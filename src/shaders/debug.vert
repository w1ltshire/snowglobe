#version 330 core

layout (location = 0) in vec2 aPos;

void main() {
    // Fixed screen size of 448x496
    // vec2 normalized_position = aPos / vec2(224.0, 248.0) * 2.0 - 1.0;
    // gl_Position = vec4(normalized_position.x, -normalized_position.y, 0.0, 1.0);
    gl_Position = vec4(aPos, 0.0, 1.0);
}