//! Residual-stream probe interface wired into llama.cpp's `cb_eval`.
//!
//! The callback fires for every graph node, before and after evaluation.
//! We only observe tensors that match a `ProbeHandler`'s `wants()` check,
//! minimising overhead.  When observing, the handler can return an
//! overridden buffer that we write back via `ggml_backend_tensor_set`.

use llama_cpp_sys_2 as sys;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};

/// Per-node metadata the handler sees before the tensor is evaluated.
/// Used to decide whether to observe this node at all.
pub struct ProbeNode<'a> {
    /// Tensor name (e.g. "l_out-26", "attn_post_norm-26").
    pub name: &'a str,
    /// Layer index parsed from the suffix, if the tensor looks like
    /// `<kind>-<layer>`.  None for tensors not associated with a layer.
    pub layer: Option<u32>,
    /// Tensor logical shape (first two dims are `[n_embd, n_tokens]` for
    /// the residual stream tensors we care about).
    pub shape: [i64; 4],
    /// ggml type id (useful for filtering to f32).
    pub dtype: sys::ggml_type,
}

/// Handler that can observe and optionally overwrite residual-stream
/// tensors during decode.
pub trait ProbeHandler: Send {
    /// Return true to request observation + (optional) override of this
    /// tensor.  Keep this cheap — called for every graph node.
    fn wants(&self, node: &ProbeNode<'_>) -> bool;

    /// Called once the tensor has been evaluated.  `data` holds the f32
    /// contents (copied from device memory).  Return `Some(replacement)`
    /// to overwrite the tensor contents in-place; the replacement must be
    /// the same length as `data`.  Return `None` to leave unchanged.
    fn observe(
        &mut self,
        node: &ProbeNode<'_>,
        data: &[f32],
    ) -> Option<Vec<f32>>;

    /// If set, the caller's `generate()` should emit this token instead
    /// of the logits argmax for the step whose decode just completed.
    /// Used for KNN token-override mode.  Default: no forcing.
    fn forced_token(&self) -> Option<i32> {
        None
    }

    /// Clear per-step transient state after `generate()` has consumed it.
    fn reset_step(&mut self) {}
}

/// Default no-op handler used when no probe is configured.
pub struct NullProbe;
impl ProbeHandler for NullProbe {
    fn wants(&self, _node: &ProbeNode<'_>) -> bool {
        false
    }
    fn observe(&mut self, _node: &ProbeNode<'_>, _data: &[f32]) -> Option<Vec<f32>> {
        None
    }
}

/// Fires its inner handler at most once per lifetime, matching the
/// Python reference impl's "fire once per query" policy.  After the
/// inner handler forces a token (or writes an override) we stop
/// observing.  Call `arm()` to re-enable for a new query.
pub struct OneShot<P: ProbeHandler> {
    inner: P,
    fired: bool,
    wrote_override_this_step: bool,
}

impl<P: ProbeHandler> OneShot<P> {
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            fired: false,
            wrote_override_this_step: false,
        }
    }

    /// Re-enable the probe.  Use between requests in a long-lived server.
    pub fn arm(&mut self) {
        self.fired = false;
    }

    pub fn is_fired(&self) -> bool {
        self.fired
    }

    pub fn inner(&self) -> &P {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }
}

impl<P: ProbeHandler> ProbeHandler for OneShot<P> {
    fn wants(&self, node: &ProbeNode<'_>) -> bool {
        !self.fired && self.inner.wants(node)
    }

    fn observe(&mut self, node: &ProbeNode<'_>, data: &[f32]) -> Option<Vec<f32>> {
        if self.fired {
            return None;
        }
        let out = self.inner.observe(node, data);
        self.wrote_override_this_step = out.is_some();
        out
    }

    fn forced_token(&self) -> Option<i32> {
        if self.fired {
            None
        } else {
            self.inner.forced_token()
        }
    }

    fn reset_step(&mut self) {
        // Decide BEFORE delegating to the inner, because inner.reset_step()
        // may clear the state we need to inspect.
        if !self.fired
            && (self.inner.forced_token().is_some() || self.wrote_override_this_step)
        {
            self.fired = true;
        }
        self.wrote_override_this_step = false;
        self.inner.reset_step();
    }
}

/// Heap-allocated wrapper passed as `cb_eval_user_data`.  Stored in a
/// `Box` so the address is stable across moves of the pipeline.
pub(crate) struct ProbeDispatch {
    pub(crate) handler: Box<dyn ProbeHandler>,
    /// Scratch buffer reused across `observe` calls to avoid reallocation.
    scratch: Vec<f32>,
}

impl ProbeDispatch {
    pub(crate) fn new(handler: Box<dyn ProbeHandler>) -> Box<Self> {
        Box::new(Self {
            handler,
            scratch: Vec::new(),
        })
    }
    pub(crate) fn handler_mut(&mut self) -> &mut dyn ProbeHandler {
        &mut *self.handler
    }
}

fn parse_layer(name: &str) -> Option<u32> {
    // Tensor names look like "l_out-26", "attn_post_norm-26", etc.
    let idx = name.rfind('-')?;
    name[idx + 1..].parse().ok()
}

/// C callback registered via `llama_context_params::cb_eval`.
///
/// `user_data` must be a valid `*mut ProbeDispatch` produced by
/// `ProbeDispatch::new(..)` and kept alive for the lifetime of the
/// owning context.
pub(crate) unsafe extern "C" fn cb_eval_trampoline(
    t: *mut sys::ggml_tensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    if user_data.is_null() || t.is_null() {
        return true;
    }
    let dispatch = &mut *(user_data as *mut ProbeDispatch);

    let name_ptr = (&(*t).name) as *const c_char;
    let name = match CStr::from_ptr(name_ptr).to_str() {
        Ok(s) => s,
        Err(_) => return true,
    };

    let node = ProbeNode {
        name,
        layer: parse_layer(name),
        shape: (*t).ne,
        dtype: (*t).type_,
    };

    if ask {
        // Keep the decision cheap — don't touch tensor data yet.
        return dispatch.handler.wants(&node);
    }

    // Post-evaluation observation.  We only get here if `wants()` returned
    // true for this node in the ask pass.
    if node.dtype != sys::GGML_TYPE_F32 {
        return true;
    }

    let nelements = sys::ggml_nelements(t) as usize;
    let nbytes = sys::ggml_nbytes(t);
    dispatch.scratch.resize(nelements, 0.0);
    sys::ggml_backend_tensor_get(
        t,
        dispatch.scratch.as_mut_ptr() as *mut c_void,
        0,
        nbytes,
    );

    if let Some(new_data) = dispatch.handler.observe(&node, &dispatch.scratch) {
        if new_data.len() == nelements {
            sys::ggml_backend_tensor_set(
                t,
                new_data.as_ptr() as *const c_void,
                0,
                nbytes,
            );
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_layer_from_name() {
        assert_eq!(parse_layer("l_out-26"), Some(26));
        assert_eq!(parse_layer("attn_post_norm-0"), Some(0));
        assert_eq!(parse_layer("attn_post_norm-33"), Some(33));
        assert_eq!(parse_layer("token_embd.weight"), None);
        assert_eq!(parse_layer("norm"), None);
    }
}
