//! Process-wide llama.cpp backend init/teardown. `llama_backend_init`
//! must be called exactly once; we guard with a `Once`.

use llama_cpp_sys_2 as sys;
use std::sync::Once;

static INIT: Once = Once::new();

pub fn ensure_initialized() {
    INIT.call_once(|| unsafe {
        sys::llama_backend_init();
    });
}
