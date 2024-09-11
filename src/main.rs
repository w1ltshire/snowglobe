use glow::HasContext;
use rapier2d::prelude::*;
use sdl3::{
    event::{Event, WindowEvent},
    keyboard::Keycode,
};
use std::{
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
        [53.0, 2.0], // 1
        [53.0, 2.0],
        [53.0, 2.0],
        [53.0, 2.0],
        [52.0, 0.0], // 5
        [52.0, 0.0],
        [56.0, 0.0], // 7
        [56.0, 0.0],
        [55.0, -1.0], // 9
        [55.0, -1.0],
        [55.0, -1.0],
        [55.0, -1.0],
        [52.0, 0.0], // 13
        [52.0, 0.0],
        [52.0, 0.0],
        [52.0, 0.0],
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
        [-15.0, -41.0], // 1
        [-15.0, -41.0],
        [-15.0, -41.0],
        [-15.0, -41.0],
        [-16.0, -44.0], // 5
        [-16.0, -44.0],
        [-16.0, -44.0],
        [-16.0, -44.0],
        [-17.0, -44.0], // 9
        [-17.0, -44.0],
        [-17.0, -44.0],
        [-17.0, -44.0],
        [-16.0, -44.0], // 13
        [-16.0, -44.0],
        [-16.0, -44.0],
        [-16.0, -44.0],
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
unsafe extern "C" fn hit_test_fn(
    _: *mut sdl3::sys::SDL_Window,
    _: *const sdl3::sys::SDL_Point,
    _: *mut std::ffi::c_void,
) -> sdl3::sys::SDL_HitTestResult {
    sdl3::sys::SDL_HitTestResult::SDL_HITTEST_DRAGGABLE
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdl = sdl3::init()?;
    let video = sdl.video()?;

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

    let niko_collider = ColliderBuilder::cuboid(
        NIKO_COLLIDER_WIDTH / 2.0 / PHYSICS_METER_PX,
        NIKO_COLLIDER_HEIGHT / 2.0 / PHYSICS_METER_PX,
    )
    .restitution(0.75)
    .build();
    collider_set.insert_with_parent(niko_collider, niko_body_handle, &mut rigid_body_set);

    let gravity = vector![0.0, 0.0];
    let integration_parameters = IntegrationParameters::default();
    let mut physics_pipeline = PhysicsPipeline::new();
    let mut island_manager = IslandManager::new();
    let mut broad_phase = DefaultBroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut impulse_joint_set = ImpulseJointSet::new();
    let mut multibody_joint_set = MultibodyJointSet::new();
    let mut ccd_solver = CCDSolver::new();

    video.gl_attr().set_stencil_size(1);

    let window = video
        .window("snowglobe ", 224 * 2, 248 * 2)
        .set_window_flags(
            sdl3::sys::SDL_WindowFlags::SDL_WINDOW_TRANSPARENT as u32
                | sdl3::sys::SDL_WindowFlags::SDL_WINDOW_ALWAYS_ON_TOP as u32,
        )
        .borderless()
        .opengl()
        .build()?;

    // set the hit test to allow the window to be dragged by the globe
    unsafe {
        let window = window.raw();
        sdl3::sys::SDL_SetWindowHitTest(window, Some(hit_test_fn), std::ptr::null_mut());
    }

    let _sdl_gl_ctx = window.gl_create_context()?;
    let gl = unsafe {
        glow::Context::from_loader_function(|s| {
            video
                .gl_get_proc_address(s)
                .map_or(std::ptr::null(), |p| p as *const _)
        })
    };
    unsafe {
        gl.enable(glow::STENCIL_TEST);
        gl.stencil_func(glow::NOTEQUAL, 1, 0xFF);
        gl.stencil_op(glow::KEEP, glow::KEEP, glow::REPLACE);

        gl.enable(glow::BLEND);
        // alpha blending
        gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
    }

    let shader = shader::Shader::new(&gl);
    let assets = assets::Assets::load(&gl);
    let vertex_array = create_vertex_buffer(&gl);

    let mut niko = Niko::default();
    let mut state = State::Stopped;

    let mut event_pump = sdl.event_pump()?;

    let mut last = Instant::now();
    let mut accum = Duration::ZERO;

    let (mut last_window_x, mut last_window_y) = window.position();

    'el: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'el,
                Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'el,
                Event::Window {
                    win_event: WindowEvent::Moved(new_x, new_y),
                    ..
                } => {
                    let diff_x = last_window_x - new_x;
                    let diff_y = last_window_y - new_y;
                    last_window_x = new_x;
                    last_window_y = new_y;

                    rigid_body_set[niko_body_handle].apply_impulse(
                        vector![diff_x as f32, diff_y as f32] / PHYSICS_METER_PX,
                        true,
                    );
                }
                _ => {}
            }
        }

        let now = Instant::now();
        let delta = now.duration_since(last);
        accum += delta;
        last = now;

        physics_pipeline.step(
            &gravity,
            &integration_parameters,
            &mut island_manager,
            &mut broad_phase,
            &mut narrow_phase,
            &mut rigid_body_set,
            &mut collider_set,
            &mut impulse_joint_set,
            &mut multibody_joint_set,
            &mut ccd_solver,
            None,
            &(),
            &(),
        );

        let niko_body = &mut rigid_body_set[niko_body_handle];
        let niko_position = *niko_body.position();

        // speed up animation based on velocity
        let anim_velocity = niko_body.linvel().norm().mul(3.0).max(1.0);
        let frame_dur = ANIM_FRAME_DUR.div_f32(anim_velocity);
        if accum >= frame_dur {
            niko.frame += 1;
            if niko.frame >= 16 {
                niko.frame = 0;
            }
            accum = Duration::ZERO;
        }

        let niko_moving = niko_body.linvel().norm() > 0.01;
        // if niko is moving, no matter what state they were in, set state to moving
        if niko_moving && !matches!(state, State::Moving { .. }) {
            state = State::Moving { max_velocity: 0.0 };
        }

        match state {
            // Nothing is really happening, so set Niko's state to Happy
            State::Stopped => niko.face = Face::Happy,
            State::Moving {
                ref mut max_velocity,
            } => {
                if *max_velocity > 0.5 {
                    niko.face = Face::Shook;
                } else {
                    niko.face = Face::Happy;
                }

                let velocity = niko_body.linvel().norm();
                if velocity > *max_velocity {
                    *max_velocity = velocity;
                }

                // looks like Niko isn't not moving anymore, so set state to JustStopped
                if !niko_moving {
                    state = State::JustStopped {
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
                    niko.face = Face::Dizzy;
                } else if max_velocity > 0.5 {
                    niko.face = Face::Shook;
                } else {
                    niko.face = Face::Happy;
                }
                *cooldown = cooldown.saturating_sub(delta);
                if *cooldown == Duration::ZERO {
                    state = State::Stopped;
                    niko_body.set_rotation(Default::default(), false);
                }
            }
        }

        unsafe {
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::STENCIL_BUFFER_BIT);

            gl.use_program(Some(shader.program));
            gl.bind_vertex_array(Some(vertex_array));

            // don't write stand to the stencil buffer
            gl.stencil_func(glow::ALWAYS, 1, 0xFF);
            gl.stencil_mask(0x00);
            // draw stand
            draw_texture_at(
                &gl,
                glam::vec2(24.0, 166.0),
                0.0,
                assets.globe.stand,
                &shader,
            );

            // draw globe mask into stencil buffer

            gl.stencil_func(glow::ALWAYS, 1, 0xFF);
            gl.stencil_mask(0xFF);
            draw_texture_at(&gl, glam::vec2(0.0, 0.0), 0.0, assets.globe.mask, &shader);

            // draw niko, ensuring that they are clipped by the stencil buffer
            gl.stencil_func(glow::EQUAL, 1, 0xFF);
            gl.stencil_mask(0x00);

            let niko_top_left =
                point![-NIKO_COLLIDER_WIDTH / 2.0, -NIKO_COLLIDER_HEIGHT / 2.0] / PHYSICS_METER_PX;
            let translated_niko_top_left = niko_position * niko_top_left * PHYSICS_METER_PX;

            {
                let texture = niko.texture(&assets);
                let tex_size = glam::vec2(texture.width as f32, texture.height as f32);

                let [niko_ox, niko_oy] = niko.frame_offset();
                let niko_offset = glam::vec2(niko_ox, niko_oy);
                let niko_tex_position =
                    glam::vec2(translated_niko_top_left.x, translated_niko_top_left.y)
                        - niko_offset;
                let niko_angle = niko_position.rotation.angle();
                let rotation_pos = niko_offset;

                let transform = glam::Affine2::from_translation(rotation_pos);
                let transform = transform * glam::Affine2::from_angle(niko_angle);
                let transform = transform * glam::Affine2::from_translation(-rotation_pos);

                let transform = transform * glam::Affine2::from_scale(tex_size);
                let transform = glam::Affine2::from_translation(niko_tex_position) * transform;

                draw_texture_affine(&gl, transform, texture, &shader);
            }

            {
                let texture = niko.face.texture(&assets);
                let tex_size = glam::vec2(texture.width as f32, texture.height as f32);

                let [face_ox, face_oy] = niko.face.offset_for(niko.frame);
                let face_offset = glam::vec2(face_ox, face_oy);
                let face_tex_position =
                    glam::vec2(translated_niko_top_left.x, translated_niko_top_left.y)
                        - face_offset;
                let face_angle = niko_position.rotation.angle();
                let rotation_pos = face_offset;

                let transform = glam::Affine2::from_translation(rotation_pos);
                let transform = transform * glam::Affine2::from_angle(face_angle);
                let transform = transform * glam::Affine2::from_translation(-rotation_pos);

                let transform = transform * glam::Affine2::from_scale(tex_size);
                let transform = glam::Affine2::from_translation(face_tex_position) * transform;

                draw_texture_affine(&gl, transform, texture, &shader);
            }

            draw_texture_at(&gl, glam::vec2(0.0, 0.0), 0.0, assets.globe.glass, &shader);

            window.gl_swap_window();

            std::thread::sleep(FRAME_DUR);
        }
    }

    Ok(())
}
