#version 330 core

in vec2 tex_coords;
out uint stencil;

uniform sampler2D u_texture;

void main() {
    vec4 sampled = texture(u_texture, tex_coords);

    if (sampled.a < 0.1) {
      discard;
    }

    stencil = 1u;
}