#version 330 core

in vec2 tex_coords;
out vec4 color;

uniform sampler2D u_texture;

void main() {
    color = texture(u_texture, tex_coords);

    if (color.a < 0.1) {
      discard;
    }
}