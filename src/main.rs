#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(clippy::not_unsafe_ptr_arg_deref)] // we have functions that are marked as pub, but are only used in C

use glow::HasContext;
use rapier2d::prelude::*;

use std::{
    ffi::{c_char, c_int, c_void},
    ops::Mul,
    time::{Duration, Instant},
};

mod raw_assets {
    pub mod globe {
        pub const GLASS: &[u8] = include_bytes!("assets/globe/glass.png");
        pub const STAND: &[u8] = include_bytes!("assets/globe/stand.png");
        pub const MASK: &[u8] = include_bytes!("assets/globe/globe_mask.png");
    }

    pub mod niko {
        pub mod faces {
            pub const DIZZY: &[u8] = include_bytes!("assets/niko/faces/dizzy.png");
            pub const HAPPY: &[u8] = include_bytes!("assets/niko/faces/happy.png");
            pub const SHOOK: &[u8] = include_bytes!("assets/niko/faces/shook.png");
        }

        // i really wish i had some kind of macro to do this
        pub const FRAMES: [&[u8]; 16] = [
            include_bytes!("assets/niko/frames/00.png"),
            include_bytes!("assets/niko/frames/01.png"),
            include_bytes!("assets/niko/frames/02.png"),
            include_bytes!("assets/niko/frames/03.png"),
            include_bytes!("assets/niko/frames/04.png"),
            include_bytes!("assets/niko/frames/05.png"),
            include_bytes!("assets/niko/frames/06.png"),
            include_bytes!("assets/niko/frames/07.png"),
            include_bytes!("assets/niko/frames/08.png"),
            include_bytes!("assets/niko/frames/09.png"),
            include_bytes!("assets/niko/frames/10.png"),
            include_bytes!("assets/niko/frames/11.png"),
            include_bytes!("assets/niko/frames/12.png"),
            include_bytes!("assets/niko/frames/13.png"),
            include_bytes!("assets/niko/frames/14.png"),
            include_bytes!("assets/niko/frames/15.png"),
        ];
    }
}

pub mod assets {
    use glow::HasContext;

    use crate::raw_assets;

    #[derive(Debug, Clone, Copy)]
    pub struct Texture {
        pub raw: glow::NativeTexture,
        pub width: u32,
        pub height: u32,
    }

    pub struct Assets {
        pub globe: Globe,
        pub niko: Niko,
    }

    pub struct Globe {
        pub glass: Texture,
        pub stand: Texture,
        pub mask: Texture,
    }

    pub struct Niko {
        pub faces: Faces,
        pub frames: [Texture; 16],
    }

    pub struct Faces {
        pub dizzy: Texture,
        pub happy: Texture,
        pub shook: Texture,
    }

    fn texture_from_bytes(gl: &glow::Context, bytes: &'static [u8]) -> Texture {
        let image = image::load_from_memory(bytes).expect("invalid image");
        let rgba_image = image.into_rgba8();
        let width = rgba_image.width();
        let height = rgba_image.height();

        let texture = unsafe { gl.create_texture() }.expect("failed to create texture");
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                width as i32,
                height as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                Some(&rgba_image),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
        }

        Texture {
            raw: texture,
            width,
            height,
        }
    }

    impl Assets {
        pub fn load(gl: &glow::Context) -> Self {
            let globe = Globe {
                glass: texture_from_bytes(gl, raw_assets::globe::GLASS),
                stand: texture_from_bytes(gl, raw_assets::globe::STAND),
                mask: texture_from_bytes(gl, raw_assets::globe::MASK),
            };

            let faces = Faces {
                dizzy: texture_from_bytes(gl, raw_assets::niko::faces::DIZZY),
                happy: texture_from_bytes(gl, raw_assets::niko::faces::HAPPY),
                shook: texture_from_bytes(gl, raw_assets::niko::faces::SHOOK),
            };

            let frames = std::array::from_fn(|i| {
                // ideally we'd be able to use an iterator and map this, but rust doesnt have that yet
                let frame = raw_assets::niko::FRAMES[i];
                texture_from_bytes(gl, frame)
            });

            let niko = Niko { faces, frames };

            Assets { globe, niko }
        }
    }
}

pub mod shader {
    use glow::HasContext;

    pub const VERTEX: &str = include_str!("shaders/basic.vert");
    pub const FRAGMENT: &str = include_str!("shaders/basic.frag");

    pub struct Shader {
        pub program: glow::NativeProgram,

        pub u_translation: glow::UniformLocation,
        pub u_matrix: glow::UniformLocation,
        pub u_texture: glow::UniformLocation,
    }

    impl Shader {
        pub fn new(gl: &glow::Context) -> Shader {
            let program = unsafe { gl.create_program() }.expect("failed to create program");

            let vert_shader = unsafe { gl.create_shader(glow::VERTEX_SHADER) }
                .expect("failed to create vertex shader");
            unsafe {
                gl.shader_source(vert_shader, VERTEX);
                gl.compile_shader(vert_shader);

                if !gl.get_shader_compile_status(vert_shader) {
                    panic!(
                        "failed to compile vertex shader: {}",
                        gl.get_shader_info_log(vert_shader)
                    );
                }

                gl.attach_shader(program, vert_shader);
            }

            let frag_shader = unsafe { gl.create_shader(glow::FRAGMENT_SHADER) }
                .expect("failed to create fragment shader");
            unsafe {
                gl.shader_source(frag_shader, FRAGMENT);
                gl.compile_shader(frag_shader);

                if !gl.get_shader_compile_status(frag_shader) {
                    panic!(
                        "failed to compile fragment shader: {}",
                        gl.get_shader_info_log(frag_shader)
                    );
                }

                gl.attach_shader(program, frag_shader);
            }

            unsafe {
                gl.link_program(program);

                if !gl.get_program_link_status(program) {
                    panic!(
                        "failed to link program: {}",
                        gl.get_program_info_log(program)
                    );
                }
            }

            let u_translation = unsafe { gl.get_uniform_location(program, "u_translation") }
                .expect("failed to get uniform location");
            let u_matrix = unsafe { gl.get_uniform_location(program, "u_matrix") }
                .expect("failed to get uniform location");
            let u_texture = unsafe { gl.get_uniform_location(program, "u_texture") }
                .expect("failed to get uniform location");

            Shader {
                program,

                u_translation,
                u_matrix,
                u_texture,
            }
        }
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct Niko {
    frame: usize, // 0..16
    face: Face,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    Dizzy,
    #[default]
    Happy,
    Shook,
}

type Offset = [f32; 2];

impl Niko {
    const OFFSETS: [Offset; 16] = [
        [48.0, 12.0], // 1 
        [48.0, 12.0],
        [48.0, 12.0],
        [48.0, 12.0],
        [47.0, 14.0], // 5
        [47.0, 14.0],
        [51.0, 14.0], // 7
        [51.0, 14.0],
        [50.0, 15.0], // 9
        [50.0, 15.0],
        [50.0, 15.0],
        [50.0, 15.0],
        [47.0, 14.0], // 13
        [47.0, 14.0],
        [47.0, 14.0],
        [47.0, 14.0],
    ];

    pub fn frame_offset(self) -> Offset {
        Self::OFFSETS[self.frame]
    }

    pub fn texture(self, assets: &assets::Assets) -> assets::Texture {
        assets.niko.frames[self.frame]
    }
}

impl Face {
    const OFFSETS: [Offset; 16] = [
        [-20.0, -27.0], // 1
        [-20.0, -27.0],
        [-20.0, -27.0],
        [-20.0, -27.0],
        [-21.0, -30.0], // 5
        [-21.0, -30.0],
        [-21.0, -30.0],
        [-21.0, -30.0],
        [-22.0, -30.0], // 9
        [-22.0, -30.0],
        [-22.0, -30.0],
        [-22.0, -30.0],
        [-21.0, -30.0], // 13
        [-21.0, -30.0],
        [-21.0, -30.0],
        [-21.0, -30.0],
    ];

    pub fn offset_for(self, frame: usize) -> Offset {
        Self::OFFSETS[frame]
    }

    pub fn texture(self, assets: &assets::Assets) -> assets::Texture {
        match self {
            Self::Happy => assets.niko.faces.happy,
            Self::Dizzy => assets.niko.faces.dizzy,
            Self::Shook => assets.niko.faces.shook,
        }
    }
}

const FRAME_DUR: Duration = Duration::from_millis(16);
const ANIM_FRAME_DUR: Duration = Duration::from_millis(100);
const PHYSICS_METER_PX: f32 = 100.0;

const NIKO_COLLIDER_WIDTH: f32 = 77.0;
const NIKO_COLLIDER_HEIGHT: f32 = 144.0;

const UPRIGHT_COOLDOWN: Duration = Duration::from_millis(1000);

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Stopped,
    Moving {
        max_velocity: f32,
    },
    // cooldown until Niko stops being dizzy and uprights themselves
    JustStopped {
        cooldown: Duration,
        max_velocity: f32,
    },
}

// the sdl3 crate does not expose the API for setting the hit test, so we have to do it using the raw bindings
// we really don't care about the window or the point, so we just return SDL_HITTEST_DRAGGABLE
extern "C" fn hit_test_fn(
    _: *mut sdl3_sys::video::SDL_Window,
    _: *const sdl3_sys::rect::SDL_Point,
    _: *mut std::ffi::c_void,
) -> sdl3_sys::video::SDL_HitTestResult {
    sdl3_sys::video::SDL_HitTestResult::DRAGGABLE
}

fn create_vertex_buffer(gl: &glow::Context) -> glow::NativeVertexArray {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Vertex {
        position: [f32; 2],
        tex_coords: [f32; 2],
    }

    // we just want a 1x1 square with full texture coords
    const VERTICES: [Vertex; 6] = [
        Vertex {
            position: [0.0, 0.0],
            tex_coords: [0.0, 0.0],
        },
        Vertex {
            position: [1.0, 0.0],
            tex_coords: [1.0, 0.0],
        },
        Vertex {
            position: [1.0, 1.0],
            tex_coords: [1.0, 1.0],
        },
        //
        Vertex {
            position: [0.0, 0.0],
            tex_coords: [0.0, 0.0],
        },
        Vertex {
            position: [1.0, 1.0],
            tex_coords: [1.0, 1.0],
        },
        Vertex {
            position: [0.0, 1.0],
            tex_coords: [0.0, 1.0],
        },
    ];

    unsafe {
        let vao = gl
            .create_vertex_array()
            .expect("failed to create vertex array");
        gl.bind_vertex_array(Some(vao));

        let buffer = gl.create_buffer().expect("failed to create buffer");
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));

        let vertices = bytemuck::cast_slice(&VERTICES);
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, vertices, glow::STATIC_DRAW);

        let stride = std::mem::size_of::<Vertex>() as i32;
        let position_offset = 0;
        let tex_coords_offset = std::mem::size_of::<[f32; 2]>() as i32;

        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, stride, position_offset);

        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, stride, tex_coords_offset);

        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        vao
    }
}

fn draw_texture_at(
    gl: &glow::Context,
    pos: glam::Vec2,
    angle: f32,
    texture: assets::Texture,
    shader: &shader::Shader,
) {
    let affine = glam::Affine2::from_scale_angle_translation(
        glam::vec2(texture.width as f32, texture.height as f32),
        angle,
        pos,
    );
    draw_texture_affine(gl, affine, texture, shader)
}

fn draw_texture_affine(
    gl: &glow::Context,
    affine: glam::Affine2,
    texture: assets::Texture,
    shader: &shader::Shader,
) {
    unsafe {
        gl.uniform_matrix_2_f32_slice(Some(&shader.u_matrix), false, affine.matrix2.as_ref());
        gl.uniform_2_f32_slice(Some(&shader.u_translation), affine.translation.as_ref());

        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture.raw));
        gl.uniform_1_i32(Some(&shader.u_texture), 0);

        gl.draw_arrays(glow::TRIANGLES, 0, 6);
    }
}

struct App {
    window: *mut sdl3_sys::video::SDL_Window,
    last_window_x: i32,
    last_window_y: i32,
    gl: glow::Context,

    shader: shader::Shader,
    assets: assets::Assets,

    vertex_array: glow::NativeVertexArray,

    niko: Niko,
    state: State,

    last: Instant,
    accum: Duration,

    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    ccd_solver: CCDSolver,

    niko_body_handle: RigidBodyHandle,
}

// i love smuggling pointers across FFI boundaries
// y...yayyyyyyy....
extern "C" fn app_init(
    appstate: *mut *mut c_void,
    _argc: c_int,
    _argv: *mut *mut c_char,
) -> sdl3_sys::init::SDL_AppResult {
    unsafe {
        sdl3_sys::init::SDL_Init(sdl3_sys::init::SDL_INIT_VIDEO | sdl3_sys::init::SDL_INIT_EVENTS)
    };

    let mut rigid_body_set = RigidBodySet::new();
    let mut collider_set = ColliderSet::new();

    let globe_vertices = vec![
        // verts are defined in pixels for convenience, but we need to convert them to meters
        point![2.0, 111.0] / PHYSICS_METER_PX,
        point![19.0, 56.0] / PHYSICS_METER_PX,
        point![63.0, 14.0] / PHYSICS_METER_PX,
        point![113.0, 1.0] / PHYSICS_METER_PX,
        point![172.0, 20.0] / PHYSICS_METER_PX,
        point![206.0, 56.0] / PHYSICS_METER_PX,
        point![220.0, 111.0] / PHYSICS_METER_PX,
        point![210.0, 157.0] / PHYSICS_METER_PX,
        point![182.0, 187.0] / PHYSICS_METER_PX,
        point![139.0, 204.0] / PHYSICS_METER_PX,
        point![107.0, 208.0] / PHYSICS_METER_PX,
        point![61.0, 199.0] / PHYSICS_METER_PX,
        point![25.0, 176.0] / PHYSICS_METER_PX,
        point![7.0, 147.0] / PHYSICS_METER_PX,
        // duplicate the first vertex so we get a full circle
        point![2.0, 111.0] / PHYSICS_METER_PX,
    ];
    let globe_collider = ColliderBuilder::polyline(globe_vertices, None).build();
    collider_set.insert(globe_collider);

    let niko_body = RigidBodyBuilder::dynamic()
        .translation(vector![111.0, 109.0] / PHYSICS_METER_PX)
        .ccd_enabled(true) // people will probably be violently shaking the globe
        .build();
    let niko_body_handle = rigid_body_set.insert(niko_body);

    let niko_collider_vertices = vec![
        point![0.0, 77.0] / PHYSICS_METER_PX,
        point![14.0, 6.0] / PHYSICS_METER_PX,
        point![55.0, 0.0] / PHYSICS_METER_PX,
        point![79.0, 66.0] / PHYSICS_METER_PX,
        point![46.0, 122.0] / PHYSICS_METER_PX,
        point![5.0, 124.0] / PHYSICS_METER_PX,
        point![0.0, 77.0] / PHYSICS_METER_PX,
    ];
    
    let niko_collider = ColliderBuilder::polyline(niko_collider_vertices, None)
    //.restitution(0.75) // No clue what this does but line 501 doesn't have it :>
    .build();
    collider_set.insert_with_parent(niko_collider, niko_body_handle, &mut rigid_body_set);

    let integration_parameters = IntegrationParameters::default();
    let physics_pipeline = PhysicsPipeline::new();
    let island_manager = IslandManager::new();
    let broad_phase = DefaultBroadPhase::new();
    let narrow_phase = NarrowPhase::new();
    let impulse_joint_set = ImpulseJointSet::new();
    let multibody_joint_set = MultibodyJointSet::new();
    let ccd_solver = CCDSolver::new();

    let window = unsafe {
        sdl3_sys::video::SDL_GL_SetAttribute(sdl3_sys::video::SDL_GLattr::STENCIL_SIZE, 1);
        sdl3_sys::video::SDL_CreateWindow(
            c"snowglobe".as_ptr(), // <3 c string literals
            224 * 2,
            248 * 2,
            sdl3_sys::video::SDL_WINDOW_OPENGL
                | sdl3_sys::video::SDL_WINDOW_TRANSPARENT
                | sdl3_sys::video::SDL_WINDOW_ALWAYS_ON_TOP
                | sdl3_sys::video::SDL_WINDOW_BORDERLESS
                | sdl3_sys::video::SDL_WINDOW_UTILITY,
        )
    };

    // set the hit test to allow the window to be dragged by the globe
    unsafe {
        sdl3_sys::video::SDL_SetWindowHitTest(window, Some(hit_test_fn), std::ptr::null_mut());
    }

    // initialize glow
    let gl = unsafe {
        sdl3_sys::video::SDL_GL_CreateContext(window);
        glow::Context::from_loader_function_cstr(|s| {
            sdl3_sys::video::SDL_GL_GetProcAddress(s.as_ptr())
                .map_or(std::ptr::null(), |p| p as *const _)
        })
    };
    unsafe {
        gl.enable(glow::STENCIL_TEST);
        gl.stencil_func(glow::NOTEQUAL, 1, 0xFF);
        gl.stencil_op(glow::KEEP, glow::KEEP, glow::REPLACE);

        gl.enable(glow::BLEND);
        // alpha blending
        gl.blend_func_separate(
            glow::SRC_ALPHA,
            glow::ONE_MINUS_SRC_ALPHA,
            glow::ONE,
            glow::ONE_MINUS_SRC_ALPHA,
        );
    }

    let shader = shader::Shader::new(&gl);
    let assets = assets::Assets::load(&gl);
    let vertex_array = create_vertex_buffer(&gl);

    let niko = Niko::default();
    let state = State::Stopped;

    let last = Instant::now();
    let accum = Duration::ZERO;

    let mut last_window_x = 0;
    let mut last_window_y = 0;
    unsafe {
        sdl3_sys::video::SDL_GetWindowPosition(window, &mut last_window_x, &mut last_window_y);
    }

    let app = App {
        window,
        last_window_x,
        last_window_y,

        gl,

        shader,
        assets,
        vertex_array,

        niko,
        state,

        last,
        accum,

        integration_parameters,
        physics_pipeline,
        island_manager,
        broad_phase,
        narrow_phase,
        impulse_joint_set,
        multibody_joint_set,
        rigid_body_set,
        collider_set,
        ccd_solver,

        niko_body_handle,
    };
    let app = Box::new(app);
    let app = Box::into_raw(app);

    unsafe {
        *appstate = app.cast::<c_void>();
    }

    sdl3_sys::init::SDL_AppResult::CONTINUE
}

extern "C" fn app_iterate(appstate: *mut c_void) -> sdl3_sys::init::SDL_AppResult {
    let app = unsafe { &mut *appstate.cast::<App>() };

    let now = Instant::now();
    let delta = now.duration_since(app.last);
    app.accum += delta;
    app.last = now;

    app.physics_pipeline.step(
        &vector![0.0, 0.0],
        &app.integration_parameters,
        &mut app.island_manager,
        &mut app.broad_phase,
        &mut app.narrow_phase,
        &mut app.rigid_body_set,
        &mut app.collider_set,
        &mut app.impulse_joint_set,
        &mut app.multibody_joint_set,
        &mut app.ccd_solver,
        None,
        &(),
        &(),
    );

    let niko_body = &mut app.rigid_body_set[app.niko_body_handle];
    let niko_position = *niko_body.position();

    // speed up animation based on velocity
    let anim_velocity = niko_body.linvel().norm().mul(3.0).max(1.0);
    let frame_dur = ANIM_FRAME_DUR.div_f32(anim_velocity);
    if app.accum >= frame_dur {
        app.niko.frame += 1;
        if app.niko.frame >= 16 {
            app.niko.frame = 0;
        }
        app.accum = Duration::ZERO;
    }

    let niko_moving = niko_body.linvel().norm() > 0.01;
    // if niko is moving, no matter what state they were in, set state to moving
    if niko_moving && !matches!(app.state, State::Moving { .. }) {
        app.state = State::Moving { max_velocity: 0.0 };
    }

    match app.state {
        // Nothing is really happening, so set Niko's state to Happy
        State::Stopped => app.niko.face = Face::Happy,
        State::Moving {
            ref mut max_velocity,
        } => {
            if *max_velocity > 0.5 {
                app.niko.face = Face::Shook;
            } else {
                app.niko.face = Face::Happy;
            }

            let velocity = niko_body.linvel().norm();
            if velocity > *max_velocity {
                *max_velocity = velocity;
            }

            // looks like Niko isn't not moving anymore, so set state to JustStopped
            if !niko_moving {
                app.state = State::JustStopped {
                    cooldown: UPRIGHT_COOLDOWN,
                    max_velocity: *max_velocity,
                };
            }
        }
        State::JustStopped {
            ref mut cooldown,
            max_velocity,
        } => {
            if max_velocity > 10.0 {
                app.niko.face = Face::Dizzy;
            } else if max_velocity > 0.5 {
                app.niko.face = Face::Shook;
            } else {
                app.niko.face = Face::Happy;
            }
            *cooldown = cooldown.saturating_sub(delta);
            if *cooldown == Duration::ZERO {
                app.state = State::Stopped;
                niko_body.set_rotation(Default::default(), false);
            }
        }
    }

    unsafe {
        app.gl.clear_color(0.0, 0.0, 0.0, 0.0);
        app.gl
            .clear(glow::COLOR_BUFFER_BIT | glow::STENCIL_BUFFER_BIT);

        app.gl.use_program(Some(app.shader.program));
        app.gl.bind_vertex_array(Some(app.vertex_array));

        // don't write stand to the stencil buffer
        app.gl.stencil_func(glow::ALWAYS, 1, 0xFF);
        app.gl.stencil_mask(0x00);
        // draw stand
        draw_texture_at(
            &app.gl,
            glam::vec2(24.0, 166.0),
            0.0,
            app.assets.globe.stand,
            &app.shader,
        );

        // draw globe mask into stencil buffer

        app.gl.stencil_func(glow::ALWAYS, 1, 0xFF);
        app.gl.stencil_mask(0xFF);
        draw_texture_at(
            &app.gl,
            glam::vec2(0.0, 0.0),
            0.0,
            app.assets.globe.mask,
            &app.shader,
        );

        // draw niko, ensuring that they are clipped by the stencil buffer
        app.gl.stencil_func(glow::EQUAL, 1, 0xFF);
        app.gl.stencil_mask(0x00);

        let niko_top_left =
            point![-NIKO_COLLIDER_WIDTH / 2.0, -NIKO_COLLIDER_HEIGHT / 2.0] / PHYSICS_METER_PX;
        let translated_niko_top_left = niko_position * niko_top_left * PHYSICS_METER_PX;

        {
            let texture = app.niko.texture(&app.assets);
            let tex_size = glam::vec2(texture.width as f32, texture.height as f32);

            let [niko_ox, niko_oy] = app.niko.frame_offset();
            let niko_offset = glam::vec2(niko_ox, niko_oy);
            let niko_tex_position =
                glam::vec2(translated_niko_top_left.x, translated_niko_top_left.y) - niko_offset;
            let niko_angle = niko_position.rotation.angle();
            let rotation_pos = niko_offset;

            let transform = glam::Affine2::from_translation(rotation_pos);
            let transform = transform * glam::Affine2::from_angle(niko_angle);
            let transform = transform * glam::Affine2::from_translation(-rotation_pos);

            let transform = transform * glam::Affine2::from_scale(tex_size);
            let transform = glam::Affine2::from_translation(niko_tex_position) * transform;

            draw_texture_affine(&app.gl, transform, texture, &app.shader);
        }

        {
            let texture = app.niko.face.texture(&app.assets);
            let tex_size = glam::vec2(texture.width as f32, texture.height as f32);

            let [face_ox, face_oy] = app.niko.face.offset_for(app.niko.frame);
            let face_offset = glam::vec2(face_ox, face_oy);
            let face_tex_position =
                glam::vec2(translated_niko_top_left.x, translated_niko_top_left.y) - face_offset;
            let face_angle = niko_position.rotation.angle();
            let rotation_pos = face_offset;

            let transform = glam::Affine2::from_translation(rotation_pos);
            let transform = transform * glam::Affine2::from_angle(face_angle);
            let transform = transform * glam::Affine2::from_translation(-rotation_pos);

            let transform = transform * glam::Affine2::from_scale(tex_size);
            let transform = glam::Affine2::from_translation(face_tex_position) * transform;

            draw_texture_affine(&app.gl, transform, texture, &app.shader);
        }

        draw_texture_at(
            &app.gl,
            glam::vec2(0.0, 0.0),
            0.0,
            app.assets.globe.glass,
            &app.shader,
        );

        sdl3_sys::video::SDL_GL_SwapWindow(app.window);

        std::thread::sleep(FRAME_DUR);
    }

    sdl3_sys::init::SDL_AppResult::CONTINUE
}

extern "C" fn app_event(
    appstate: *mut c_void,
    event: *mut sdl3_sys::events::SDL_Event,
) -> sdl3_sys::init::SDL_AppResult {
    unsafe {
        let state = &mut *appstate.cast::<App>();
        let event = *event;
        match sdl3_sys::events::SDL_EventType(event.r#type as i32) {
            sdl3_sys::events::SDL_EventType::QUIT => sdl3_sys::init::SDL_AppResult::FAILURE,
            sdl3_sys::events::SDL_EventType::KEY_DOWN => {
                let key = event.key.key;
                if key == sdl3_sys::keycode::SDLK_ESCAPE {
                    sdl3_sys::init::SDL_AppResult::FAILURE
                } else {
                    sdl3_sys::init::SDL_AppResult::CONTINUE
                }
            }
            sdl3_sys::events::SDL_EventType::WINDOW_MOVED => {
                let new_x = event.window.data1;
                let new_y = event.window.data2;
                let diff_x = state.last_window_x - new_x;
                let diff_y = state.last_window_y - new_y;
                state.last_window_x = new_x;
                state.last_window_y = new_y;

                state.rigid_body_set[state.niko_body_handle].apply_impulse(
                    vector![diff_x as f32, diff_y as f32] / PHYSICS_METER_PX,
                    true,
                );
                sdl3_sys::init::SDL_AppResult::CONTINUE
            }
            _ => sdl3_sys::init::SDL_AppResult::CONTINUE,
        }
    }
}

extern "C" fn app_quit(appstate: *mut c_void) {
    unsafe {
        let app = Box::from_raw(appstate.cast::<App>());
        sdl3_sys::video::SDL_DestroyWindow(app.window);
        sdl3_sys::init::SDL_Quit();
    }
}

// sdl3_sys doesn't provide bindings for SDL_main.h, so we have to fudge this a little
extern "C" {
    fn SDL_EnterAppMainCallbacks(
        _argc: c_int,
        _argv: *mut *mut c_char,
        app_init: sdl3_sys::init::SDL_AppInit_func,
        app_iterate: sdl3_sys::init::SDL_AppIterate_func,
        app_event: sdl3_sys::init::SDL_AppEvent_func,
        app_quit: sdl3_sys::init::SDL_AppQuit_func,
    ) -> c_int;
}

fn main() {
    // normally SDL_main would generate a main function that would call this for us, but that's kiiiiiiiinda not possible
    unsafe {
        SDL_EnterAppMainCallbacks(
            0,
            std::ptr::null_mut(),
            Some(app_init),
            Some(app_iterate),
            Some(app_event),
            Some(app_quit),
        );
    }
}
