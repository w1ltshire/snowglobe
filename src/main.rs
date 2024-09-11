use rapier2d::prelude::*;
use sdl3::{
    event::{Event, WindowEvent},
    keyboard::Keycode,
    render::{Canvas, FPoint, FRect, RenderTarget, Texture},
};
use std::{
    ops::Mul,
    time::{Duration, Instant},
};

mod raw_assets {
    pub mod globe {
        pub const GLASS: &[u8] = include_bytes!("assets/globe/glass.png");
        pub const STAND: &[u8] = include_bytes!("assets/globe/stand.png");
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
    use sdl3::{
        render::{Texture, TextureCreator},
        surface::Surface,
    };

    use crate::raw_assets;

    pub struct Assets<'tex> {
        pub globe: Globe<'tex>,
        pub niko: Niko<'tex>,
    }

    pub struct Globe<'tex> {
        pub glass: Texture<'tex>,
        pub stand: Texture<'tex>,
    }

    pub struct Niko<'tex> {
        pub faces: Faces<'tex>,
        pub frames: [Texture<'tex>; 16],
    }

    pub struct Faces<'tex> {
        pub dizzy: Texture<'tex>,
        pub happy: Texture<'tex>,
        pub shook: Texture<'tex>,
    }

    fn sdl_texture_from_bytes<'tex, T>(
        creator: &'tex TextureCreator<T>,
        bytes: &'static [u8],
    ) -> Texture<'tex> {
        let image = image::load_from_memory(bytes).expect("invalid image");
        let mut rgba_image = image.into_rgba8();
        let width = rgba_image.width();
        let height = rgba_image.height();

        let surface = Surface::from_data(
            &mut rgba_image,
            width,
            height,
            width * 4,
            sdl3::pixels::PixelFormatEnum::ABGR8888,
        )
        .expect("failed to create surface");
        surface
            .as_texture(creator)
            .expect("failed to convert texture to surface")
    }

    impl<'tex> Assets<'tex> {
        pub fn load<T>(creator: &'tex TextureCreator<T>) -> Self {
            let globe = Globe {
                glass: sdl_texture_from_bytes(creator, raw_assets::globe::GLASS),
                stand: sdl_texture_from_bytes(creator, raw_assets::globe::STAND),
            };

            let faces = Faces {
                dizzy: sdl_texture_from_bytes(creator, raw_assets::niko::faces::DIZZY),
                happy: sdl_texture_from_bytes(creator, raw_assets::niko::faces::HAPPY),
                shook: sdl_texture_from_bytes(creator, raw_assets::niko::faces::SHOOK),
            };

            let frames = std::array::from_fn(|i| {
                // ideally we'd be able to use an iterator and map this, but rust doesnt have that yet
                let frame = raw_assets::niko::FRAMES[i];
                sdl_texture_from_bytes(creator, frame)
            });

            let niko = Niko { faces, frames };

            Assets { globe, niko }
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

    pub fn texture<'a, 'tex>(self, assets: &'a assets::Assets<'tex>) -> &'a Texture<'tex> {
        &assets.niko.frames[self.frame]
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

    pub fn texture<'a, 'tex>(self, assets: &'a assets::Assets<'tex>) -> &'a Texture<'tex> {
        match self {
            Self::Happy => &assets.niko.faces.happy,
            Self::Dizzy => &assets.niko.faces.dizzy,
            Self::Shook => &assets.niko.faces.shook,
        }
    }
}

const FRAME_DUR: Duration = Duration::from_millis(16);
const ANIM_FRAME_DUR: Duration = Duration::from_millis(100);
const PHYSICS_METER_PX: f32 = 100.0;

fn draw_texture_at<T: RenderTarget>(offset: Offset, canvas: &mut Canvas<T>, texture: &Texture<'_>) {
    let query = texture.query();
    let src = FRect::new(0.0, 0.0, query.width as f32, query.height as f32);
    let dst = FRect::new(offset[0], offset[1], src.w, src.h);
    canvas
        .copy(texture, src, dst)
        .expect("failed to copy texture");
}

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

    let window = video
        .window("snowglobe ", 224 * 2, 248 * 2)
        .set_window_flags(
            sdl3::sys::SDL_WindowFlags::SDL_WINDOW_TRANSPARENT as u32
                | sdl3::sys::SDL_WindowFlags::SDL_WINDOW_ALWAYS_ON_TOP as u32,
        )
        .borderless()
        .build()?;

    // set the hit test to allow the window to be dragged by the globe
    unsafe {
        let window = window.raw();
        sdl3::sys::SDL_SetWindowHitTest(window, Some(hit_test_fn), std::ptr::null_mut());
    }

    let mut canvas = window.into_canvas().present_vsync().build()?;
    canvas.set_scale(2.0, 2.0)?;

    let texture_creator = canvas.texture_creator();

    let assets = assets::Assets::load(&texture_creator);
    let mut niko = Niko::default();
    let mut state = State::Stopped;

    let mut event_pump = sdl.event_pump()?;

    let mut last = Instant::now();
    let mut accum = Duration::ZERO;

    let (mut last_window_x, mut last_window_y) = canvas.window().position();

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

        canvas.set_draw_color(sdl3::pixels::Color::RGBA(0, 0, 0, 0));
        canvas.clear();

        draw_texture_at([24.0, 166.0], &mut canvas, &assets.globe.stand);

        #[cfg(debug_assertions)]
        {
            canvas.set_draw_color(sdl3::pixels::Color::RGBA(255, 0, 0, 128));
            // draw a rotated debug rect of niko
            let points = [
                // top
                point![-NIKO_COLLIDER_WIDTH / 2.0, -NIKO_COLLIDER_HEIGHT / 2.0] / PHYSICS_METER_PX,
                point![NIKO_COLLIDER_WIDTH / 2.0, -NIKO_COLLIDER_HEIGHT / 2.0] / PHYSICS_METER_PX,
                // right
                point![NIKO_COLLIDER_WIDTH / 2.0, -NIKO_COLLIDER_HEIGHT / 2.0] / PHYSICS_METER_PX,
                point![NIKO_COLLIDER_WIDTH / 2.0, NIKO_COLLIDER_HEIGHT / 2.0] / PHYSICS_METER_PX,
                // bottom
                point![NIKO_COLLIDER_WIDTH / 2.0, NIKO_COLLIDER_HEIGHT / 2.0] / PHYSICS_METER_PX,
                point![-NIKO_COLLIDER_WIDTH / 2.0, NIKO_COLLIDER_HEIGHT / 2.0] / PHYSICS_METER_PX,
                // left
                point![-NIKO_COLLIDER_WIDTH / 2.0, NIKO_COLLIDER_HEIGHT / 2.0] / PHYSICS_METER_PX,
                point![-NIKO_COLLIDER_WIDTH / 2.0, -NIKO_COLLIDER_HEIGHT / 2.0] / PHYSICS_METER_PX,
            ];
            let points: [FPoint; 8] = std::array::from_fn(|i| {
                let point = niko_position * points[i] * PHYSICS_METER_PX;
                FPoint::new(point.x, point.y)
            });
            canvas.draw_lines(points.as_slice())?;
        }

        let niko_top_left =
            point![-NIKO_COLLIDER_WIDTH / 2.0, -NIKO_COLLIDER_HEIGHT / 2.0] / PHYSICS_METER_PX;
        let translated_niko_top_left = niko_position * niko_top_left * PHYSICS_METER_PX;

        let [niko_ox, niko_oy] = niko.frame_offset();
        let texture = niko.texture(&assets);
        let query = texture.query();

        let dst = FRect::new(
            translated_niko_top_left.x - niko_ox,
            translated_niko_top_left.y - niko_oy,
            query.width as f32,
            query.height as f32,
        );
        let angle = niko_position.rotation.angle().to_degrees();
        let center = FPoint::new(niko_ox, niko_oy);

        canvas.copy_ex(texture, None, dst, angle as f64, Some(center), false, false)?;

        let [face_ox, face_oy] = niko.face.offset_for(niko.frame);
        let texture = niko.face.texture(&assets);
        let query = texture.query();

        let dst = FRect::new(
            translated_niko_top_left.x - face_ox,
            translated_niko_top_left.y - face_oy,
            query.width as f32,
            query.height as f32,
        );
        let angle = niko_position.rotation.angle().to_degrees();
        let center = FPoint::new(face_ox, face_oy);

        canvas.copy_ex(texture, None, dst, angle as f64, Some(center), false, false)?;

        draw_texture_at([0.0, 0.0], &mut canvas, &assets.globe.glass);

        canvas.present();

        std::thread::sleep(FRAME_DUR);
    }

    Ok(())
}
