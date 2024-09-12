#![allow(unused_imports, dead_code, unused_variables)]

// mostly copied from https://docs.rs/crate/sdl3-sys/0.0.7+sdl3-dev-2023-10-08/source/build.rs, but with some things removed
extern crate cmake;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs, io};

// compile a shared or static lib depending on the feature
fn compile_sdl3(sdl3_build_path: &Path, target_os: &str) -> PathBuf {
    let mut cfg = cmake::Config::new(sdl3_build_path);
    if let Ok(profile) = env::var("SDL3_BUILD_PROFILE") {
        cfg.profile(&profile);
    } else {
        cfg.profile("release");
    }

    // Allow specifying custom toolchain specifically for SDL3.
    if let Ok(toolchain) = env::var("SDL3_TOOLCHAIN") {
        cfg.define("CMAKE_TOOLCHAIN_FILE", &toolchain);
    } else {
        // Override __FLTUSED__ to keep the _fltused symbol from getting defined in the static build.
        // This conflicts and fails to link properly when building statically on Windows, likely due to
        // COMDAT conflicts/breakage happening somewhere.

        cfg.cflag("-D__FLTUSED__");
    }

    if target_os == "windows-gnu" {
        cfg.define("VIDEO_OPENGLES", "OFF");
    }

    cfg.define("SDL_SHARED", "OFF");
    cfg.define("SDL_STATIC", "ON");
    // Prevent SDL to provide it own "main" which cause a conflict when this crate linked
    // to C/C++ program.
    cfg.define("SDL_MAIN_HANDLED", "ON");

    cfg.build()
}

fn link_sdl3(target_os: &str) {
    {
        if target_os.contains("windows") && !target_os.contains("windows-gnu") {
            println!("cargo:rustc-link-lib=static=SDL3-static");
        } else {
            println!("cargo:rustc-link-lib=static=SDL3");
        }

        // Also linked to any required libraries for each supported platform
        if target_os.contains("windows") {
            println!("cargo:rustc-link-lib=shell32");
            println!("cargo:rustc-link-lib=user32");
            println!("cargo:rustc-link-lib=gdi32");
            println!("cargo:rustc-link-lib=winmm");
            println!("cargo:rustc-link-lib=imm32");
            println!("cargo:rustc-link-lib=ole32");
            println!("cargo:rustc-link-lib=oleaut32");
            println!("cargo:rustc-link-lib=version");
            println!("cargo:rustc-link-lib=uuid");
            println!("cargo:rustc-link-lib=dinput8");
            println!("cargo:rustc-link-lib=dxguid");
            println!("cargo:rustc-link-lib=setupapi");
        } else if target_os == "darwin" {
            println!("cargo:rustc-link-lib=framework=Cocoa");
            println!("cargo:rustc-link-lib=framework=IOKit");
            println!("cargo:rustc-link-lib=framework=Carbon");
            println!("cargo:rustc-link-lib=framework=ForceFeedback");
            println!("cargo:rustc-link-lib=framework=GameController");
            println!("cargo:rustc-link-lib=framework=CoreHaptics");
            println!("cargo:rustc-link-lib=framework=CoreVideo");
            println!("cargo:rustc-link-lib=framework=CoreAudio");
            println!("cargo:rustc-link-lib=framework=AudioToolbox");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=iconv");
        } else if target_os == "android" {
            println!("cargo:rustc-link-lib=android");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=GLESv1_CM");
            println!("cargo:rustc-link-lib=GLESv2");
            println!("cargo:rustc-link-lib=hidapi");
            println!("cargo:rustc-link-lib=log");
            println!("cargo:rustc-link-lib=OpenSLES");
        } else {
            // TODO: Add other platform linker options here.
        }
    }
}

fn find_cargo_target_dir() -> PathBuf {
    // Infer the top level cargo target dir from the OUT_DIR by searching
    // upwards until we get to $CARGO_TARGET_DIR/build/ (which is always one
    // level up from the deepest directory containing our package name)
    let pkg_name = env::var("CARGO_PKG_NAME").unwrap();
    let mut out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    loop {
        {
            let final_path_segment = out_dir.file_name().unwrap();
            if final_path_segment.to_string_lossy().contains(&pkg_name) {
                break;
            }
        }
        if !out_dir.pop() {
            panic!("Malformed build path: {}", out_dir.to_string_lossy());
        }
    }
    out_dir.pop();
    out_dir.pop();
    out_dir
}

#[cfg(unix)]
fn copy_library_symlink(src_path: &Path, target_path: &Path) {
    if let Ok(link_path) = fs::read_link(src_path) {
        // Copy symlinks to:
        //  * target dir: as a product ship product of the build,
        //  * deps directory: as comment example testing doesn't pick up the library search path
        //    otherwise and fails.
        let deps_path = target_path.join("deps");
        for path in &[target_path, &deps_path] {
            let dst_path = path.join(src_path.file_name().expect("Path missing filename"));
            // Silently drop errors here, in case the symlink already exists.
            let _ = std::os::unix::fs::symlink(&link_path, &dst_path);
        }
    }
}

#[cfg(not(unix))]
fn copy_library_symlink(src_path: &Path, target_path: &Path) {}

fn copy_library_file(src_path: &Path, target_path: &Path) {
    // Copy the shared libs to:
    //  * target dir: as a product ship product of the build,
    //  * deps directory: as comment example testing doesn't pick up the library search path
    //    otherwise and fails.
    let deps_path = target_path.join("deps");
    for path in &[target_path, &deps_path] {
        let dst_path = path.join(src_path.file_name().expect("Path missing filename"));

        fs::copy(src_path, &dst_path).unwrap_or_else(|_| {
            panic!(
                "Failed to copy SDL3 dynamic library from {} to {}",
                src_path.to_string_lossy(),
                dst_path.to_string_lossy()
            )
        });
    }
}

fn copy_dynamic_libraries(sdl3_compiled_path: PathBuf, target_os: &str) {
    let target_path = find_cargo_target_dir();

    // Windows binaries do not embed library search paths, so successfully
    // linking the DLL isn't sufficient to find it at runtime -- it must be
    // either on PATH or in the current working directory when we run binaries
    // linked against it. In other words, to run the test suite we need to
    // copy sdl3.dll out of its build tree and down to the top level cargo
    // binary output directory.
    if target_os.contains("windows") {
        let sdl3_dll_name = "SDL3.dll";
        let sdl3_bin_path = sdl3_compiled_path.join("bin");
        let src_dll_path = sdl3_bin_path.join(sdl3_dll_name);

        copy_library_file(&src_dll_path, &target_path);
    } else if target_os != "emscripten" {
        // Find all libraries build and copy them, symlinks included.
        let mut found = false;
        let lib_dirs = &["lib", "lib64"];
        for lib_dir in lib_dirs {
            let lib_path = sdl3_compiled_path.join(lib_dir);
            if lib_path.exists() {
                found = true;
                for entry in std::fs::read_dir(&lib_path)
                    .unwrap_or_else(|_| panic!("Couldn't readdir {}", lib_dir))
                {
                    let entry = entry.expect("Error looking at lib dir");
                    if let Ok(file_type) = entry.file_type() {
                        if file_type.is_symlink() {
                            copy_library_symlink(&entry.path(), &target_path);
                        } else if file_type.is_file() {
                            copy_library_file(&entry.path(), &target_path)
                        }
                    }
                }
                break;
            }
        }
        if !found {
            panic!("Failed to find CMake output dir");
        }
    }
}

fn main() {
    let target = env::var("TARGET").expect("Cargo build scripts always have TARGET");
    let host = env::var("HOST").expect("Cargo build scripts always have HOST");
    let target_os = get_os_from_triple(target.as_str()).unwrap();

    let sdl3_source_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("SDL");

    let sdl3_compiled_path: PathBuf;

    {
        sdl3_compiled_path = compile_sdl3(sdl3_source_path.as_path(), target_os);

        println!(
            "cargo:rustc-link-search={}",
            sdl3_compiled_path.join("lib64").display()
        );
        println!(
            "cargo:rustc-link-search={}",
            sdl3_compiled_path.join("lib").display()
        );
    }

    let sdl3_includes = sdl3_source_path
        .join("include")
        .to_str()
        .unwrap()
        .to_string();

    // we have to build and link main.c to actually provide the main() function
    cc::Build::new()
        .file("src/main.c")
        .include(sdl3_includes)
        .compile("main");
    println!("cargo:rerun-if-changed=src/main.c");

    link_sdl3(target_os);
}

fn get_os_from_triple(triple: &str) -> Option<&str> {
    triple.splitn(3, "-").nth(2)
}
