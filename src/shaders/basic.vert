#version 330 core

layout(location = 0) in vec2 m_position;
layout(location = 1) in vec2 m_tex_coords;

out vec2 tex_coords;

uniform vec2 u_translation;
uniform mat2 u_matrix;

void main() {
    vec2 calculated_position = u_matrix * m_position + u_translation;
    // we have a fixed screen size of 448x496, so we need to convert the calculated position to normalized device coordinates
    vec2 normalized_position = calculated_position / vec2(448.0, 496.0) * 2.0 - 1.0;
    gl_Position = vec4(normalized_position.x, -normalized_position.y, 0.0, 1.0);
    tex_coords = m_tex_coords;
}