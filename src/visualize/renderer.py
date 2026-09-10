"""OpenGL particle renderer and compact information dashboard."""

import bisect
import itertools
import math

import moderngl
import numpy as np

from .data import ARROW_COUNT

WIDTH, HEIGHT, FPS, FRAME_COUNT = 3840, 2160, 60, 960
SCALE = 2

PARTICLE_VERTEX = r"""
#version 430
layout(location=0) in vec4 p0;
layout(location=1) in vec4 p1;
layout(location=2) in vec4 s0;
layout(location=3) in vec4 s1;
layout(location=4) in vec2 q0;
layout(location=5) in vec2 q1;
uniform mat4 mvp;
uniform float mix_value;
uniform vec2 value_range;
uniform float point_scale;
uniform float view_radius;
uniform int slice_mode;
uniform int quantity;
out vec3 colour;
out float visibility;
out float smoothing_length;
out float particle_volume;
vec3 turbo(float x) {
    vec4 v=vec4(1,x,x*x,x*x*x); vec2 w=v.zw*v.z;
    return vec3(
        dot(v,vec4(.13572138,4.6153926,-42.660323,132.13112))+dot(w,vec2(-152.9424,59.28638)),
        dot(v,vec4(.09140261,2.1941884,4.8429666,-14.185033))+dot(w,vec2(4.2772985,2.829566)),
        dot(v,vec4(.1066733,12.641946,-60.58205,110.36277))+dot(w,vec2(-89.90311,27.34825)));
}
void main() {
    vec3 physical=mix(p0.xyz,p1.xyz,mix_value);
    vec3 display=vec3(physical.x,physical.z,physical.y);
    gl_Position=mvp*vec4(display,1);
    smoothing_length=mix(p0.w,p1.w,mix_value);
    particle_volume=mix(q0.x,q1.x,mix_value);
    visibility=step(length(physical),view_radius);
    visibility*=slice_mode==0 ? 1.0 : step(abs(physical.y),2.0*smoothing_length);
    float depth=slice_mode==0 ? 1.0/max(gl_Position.w,.01) : 1.0;
    gl_PointSize=clamp(4.0*smoothing_length*point_scale*depth,1.0,64.0);
    float magnetic=mix(log(max(s0.x,1e-30)),log(max(s1.x,1e-30)),mix_value);
    float temperature=mix(s0.y,s1.y,mix_value);
    float density=mix(log(max(q0.y,1e-30)),log(max(q1.y,1e-30)),mix_value);
    float value=quantity==0 ? magnetic : quantity==1 ? temperature : density;
    colour=turbo(clamp((value-value_range.x)/(value_range.y-value_range.x),0.0,1.0));
}
"""

PARTICLE_FRAGMENT = r"""
#version 430
in vec3 colour;
in float visibility;
in float smoothing_length;
in float particle_volume;
uniform int surface_mode;
layout(location=0) out vec4 accumulation;
layout(location=1) out float weight_sum;
void main() {
    float q=2.0*length(gl_PointCoord*2.0-1.0);
    float kernel=q<1.0 ? 1.0-1.5*q*q+.75*q*q*q : .25*pow(max(2.0-q,0.0),3.0);
    float coverage=visibility*particle_volume*kernel/pow(smoothing_length,3.0);
    float weight=surface_mode==1 ? coverage*exp(-12.0*gl_FragCoord.z) : coverage;
    if(weight<=0.0) discard;
    accumulation=vec4(colour*weight,surface_mode==1 ? coverage : weight);
    weight_sum=weight;
}
"""

SCREEN_VERTEX = r"""
#version 430
const vec2 p[6]=vec2[6](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
out vec2 uv;
void main(){gl_Position=vec4(p[gl_VertexID],0,1);uv=p[gl_VertexID]*.5+.5;}
"""

RESOLVE_FRAGMENT = r"""
#version 430
uniform sampler2D accumulated;
uniform sampler2D weights;
uniform float view_radius;
uniform float aspect;
uniform float grid_spacing;
uniform int show_grid;
in vec2 uv;
out vec4 output_colour;
void main(){
    vec4 a=texture(accumulated,uv); float w=texture(weights,uv).r;
    float opacity=1.0-exp(-a.a);
    vec3 colour=mix(vec3(.012,.018,.035),a.rgb/max(w,1e-6),opacity);
    vec2 physical=(uv*2.0-1.0)*vec2(view_radius*aspect,view_radius);
    vec2 d=abs(fract(physical/grid_spacing+.5)-.5)*grid_spacing;
    float pixel=2.0*view_radius/float(textureSize(accumulated,0).y);
    float grid=1.0-smoothstep(pixel,2.0*pixel,min(d.x,d.y));
    float axis=1.0-smoothstep(pixel,2.0*pixel,min(abs(physical.x),abs(physical.y)));
    colour=mix(colour,vec3(.45,.32,.08),float(show_grid)*.10*grid);
    colour=mix(colour,vec3(.62,.44,.10),float(show_grid)*.18*axis);
    output_colour=vec4(colour,1);
}
"""

BLIT_FRAGMENT = r"""
#version 430
uniform sampler2D source;
uniform int flip_y;
in vec2 uv;
out vec4 output_colour;
void main(){vec2 p=flip_y==1?vec2(uv.x,1.0-uv.y):uv;output_colour=texture(source,p);}
"""

UI_VERTEX = r"""
#version 430
layout(location=0) in vec2 position;
layout(location=1) in vec4 input_colour;
out vec4 colour;
void main(){gl_Position=vec4(position,0,1);colour=input_colour;}
"""

UI_FRAGMENT = r"""
#version 430
in vec4 colour;
out vec4 output_colour;
void main(){output_colour=colour;}
"""

ARROW_VERTEX = r"""
#version 430
layout(location=0) in vec4 p0;
layout(location=1) in vec4 p1;
layout(location=2) in vec4 s0;
layout(location=3) in vec4 s1;
uniform float mix_value;
uniform vec2 view_scale;
uniform vec2 magnetic_range;
out vec2 arrow_position;
out vec2 arrow_direction;
void main(){
    vec3 p=mix(p0.xyz,p1.xyz,mix_value); vec2 b=mix(s0.zw,s1.zw,mix_value);
    float m=mix(s0.x,s1.x,mix_value);
    float strength=clamp((log(max(m,1e-30))-magnetic_range.x)/(magnetic_range.y-magnetic_range.x),0.0,1.0);
    arrow_position=vec2(p.x,p.z)/view_scale;
    arrow_direction=step(1e-30,m)*normalize(b+vec2(1e-30))*(.012+.028*strength);
}
"""

ARROW_GEOMETRY = r"""
#version 430
layout(points) in;
layout(line_strip,max_vertices=6) out;
in vec2 arrow_position[];
in vec2 arrow_direction[];
void segment(vec2 a,vec2 b){gl_Position=vec4(a,0,1);EmitVertex();gl_Position=vec4(b,0,1);EmitVertex();EndPrimitive();}
void main(){vec2 p=arrow_position[0],d=arrow_direction[0],n=vec2(-d.y,d.x),tip=p+d;segment(p,tip);segment(tip,tip-d*.32+n*.24);segment(tip,tip-d*.32-n*.24);}
"""

ARROW_FRAGMENT = r"""
#version 430
uniform vec4 arrow_colour;
out vec4 output_colour;
void main(){output_colour=arrow_colour;}
"""

FONT_DATA = {
    " ": "00000/00000/00000/00000/00000/00000/00000",
    "-": "00000/00000/00000/11111/00000/00000/00000",
    ".": "00000/00000/00000/00000/00000/01100/01100",
    ":": "00000/01100/01100/00000/01100/01100/00000",
    "+": "00000/00100/00100/11111/00100/00100/00000",
    "/": "00001/00010/00100/01000/10000/00000/00000",
    "0": "01110/10001/10011/10101/11001/10001/01110",
    "1": "00100/01100/00100/00100/00100/00100/01110",
    "2": "01110/10001/00001/00010/00100/01000/11111",
    "3": "11110/00001/00001/01110/00001/00001/11110",
    "4": "00010/00110/01010/10010/11111/00010/00010",
    "5": "11111/10000/10000/11110/00001/00001/11110",
    "6": "01110/10000/10000/11110/10001/10001/01110",
    "7": "11111/00001/00010/00100/01000/01000/01000",
    "8": "01110/10001/10001/01110/10001/10001/01110",
    "9": "01110/10001/10001/01111/00001/00001/01110",
    "A": "01110/10001/10001/11111/10001/10001/10001",
    "B": "11110/10001/10001/11110/10001/10001/11110",
    "C": "01111/10000/10000/10000/10000/10000/01111",
    "D": "11110/10001/10001/10001/10001/10001/11110",
    "E": "11111/10000/10000/11110/10000/10000/11111",
    "F": "11111/10000/10000/11110/10000/10000/10000",
    "G": "01111/10000/10000/10111/10001/10001/01111",
    "H": "10001/10001/10001/11111/10001/10001/10001",
    "I": "01110/00100/00100/00100/00100/00100/01110",
    "J": "00001/00001/00001/00001/10001/10001/01110",
    "K": "10001/10010/10100/11000/10100/10010/10001",
    "L": "10000/10000/10000/10000/10000/10000/11111",
    "M": "10001/11011/10101/10101/10001/10001/10001",
    "N": "10001/11001/10101/10011/10001/10001/10001",
    "O": "01110/10001/10001/10001/10001/10001/01110",
    "P": "11110/10001/10001/11110/10000/10000/10000",
    "Q": "01110/10001/10001/10001/10101/10010/01101",
    "R": "11110/10001/10001/11110/10100/10010/10001",
    "S": "01111/10000/10000/01110/00001/00001/11110",
    "T": "11111/00100/00100/00100/00100/00100/00100",
    "U": "10001/10001/10001/10001/10001/10001/01110",
    "V": "10001/10001/10001/10001/10001/01010/00100",
    "W": "10001/10001/10001/10101/10101/10101/01010",
    "X": "10001/10001/01010/00100/01010/10001/10001",
    "Y": "10001/10001/01010/00100/00100/00100/00100",
    "Z": "11111/00001/00010/00100/01000/10000/11111",
}
FONT = {character: rows.split("/") for character, rows in FONT_DATA.items()}


def perspective(fov, aspect, near, far):
    scale = 1.0 / math.tan(math.radians(fov) / 2.0)
    return np.array(
        (
            (scale / aspect, 0, 0, 0),
            (0, scale, 0, 0),
            (0, 0, (far + near) / (near - far), 2 * far * near / (near - far)),
            (0, 0, -1, 0),
        ),
        dtype="f4",
    )


def look_at(eye, target, up):
    eye, target, up = (np.asarray(value, dtype="f4") for value in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    actual_up = np.cross(side, forward)
    result = np.eye(4, dtype="f4")
    result[:3, :3] = np.stack((side, actual_up, -forward))
    result[:3, 3] = -result[:3, :3] @ eye
    return result


def orthographic(scale, aspect, near, far):
    right = scale * aspect
    return np.array(
        (
            (1 / right, 0, 0, 0),
            (0, 1 / scale, 0, 0),
            (0, 0, -2 / (far - near), -(far + near) / (far - near)),
            (0, 0, 0, 1),
        ),
        dtype="f4",
    )


def rectangle(x0, y0, x1, y1, colour):
    return [
        (x0, y0, *colour),
        (x1, y0, *colour),
        (x1, y1, *colour),
        (x0, y0, *colour),
        (x1, y1, *colour),
        (x0, y1, *colour),
    ]


def line_quad(a, b, width, colour):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    direction = b - a
    normal = np.array((-direction[1], direction[0]))
    normal *= width / (2 * np.linalg.norm(normal))
    p0, p1, p2, p3 = a + normal, b + normal, b - normal, a - normal
    return [
        (p0[0], p0[1], *colour),
        (p1[0], p1[1], *colour),
        (p2[0], p2[1], *colour),
        (p0[0], p0[1], *colour),
        (p2[0], p2[1], *colour),
        (p3[0], p3[1], *colour),
    ]


def text_vertices(text, x, y, pixel, colour):
    vertices = []
    for character in text.upper():
        for row, bits in enumerate(FONT[character]):
            for column, bit in enumerate(bits):
                if bit == "1":
                    x0, y0 = x + column * pixel, y - row * pixel
                    vertices.extend(rectangle(x0, y0, x0 + pixel, y0 - pixel, colour))
        x += pixel * 6
    return vertices


def turbo(value):
    red = 0.13572138 + 4.61539260 * value - 42.66032258 * value**2 + 132.13108234 * value**3 - 152.94239396 * value**4 + 59.28637943 * value**5
    green = 0.09140261 + 2.19418839 * value + 4.84296658 * value**2 - 14.18503333 * value**3 + 4.27729857 * value**4 + 2.82956604 * value**5
    blue = 0.10667330 + 12.64194608 * value - 60.58204836 * value**2 + 110.36276771 * value**3 - 89.90310912 * value**4 + 27.34824973 * value**5
    return np.clip((red, green, blue), 0, 1)


class Renderer:
    def __init__(self, context, frames, name):
        self.context, self.frames = context, frames
        self.name = "".join(character if character in FONT else " " for character in name.upper())
        self.program = context.program(vertex_shader=PARTICLE_VERTEX, fragment_shader=PARTICLE_FRAGMENT)
        self.resolve = context.program(vertex_shader=SCREEN_VERTEX, fragment_shader=RESOLVE_FRAGMENT)
        self.blit = context.program(vertex_shader=SCREEN_VERTEX, fragment_shader=BLIT_FRAGMENT)
        self.ui = context.program(vertex_shader=UI_VERTEX, fragment_shader=UI_FRAGMENT)
        self.arrow = context.program(
            vertex_shader=ARROW_VERTEX,
            geometry_shader=ARROW_GEOMETRY,
            fragment_shader=ARROW_FRAGMENT,
        )
        self.final_texture = context.texture((WIDTH, HEIGHT), 4, dtype="f1")
        self.final = context.framebuffer([self.final_texture])
        self.encoding_texture = context.texture((WIDTH, HEIGHT), 4, dtype="f1")
        self.encoding = context.framebuffer([self.encoding_texture])
        self.main_panel = self._panel((1280 * SCALE, 1080 * SCALE))
        self.panels = [self._panel((640 * SCALE, 360 * SCALE)) for _ in range(2)]
        self.resolve_vao = context.vertex_array(self.resolve, [])
        self.blit_vao = context.vertex_array(self.blit, [])
        self.pair = None
        self.pair_vaos = {}
        self.static_buffer, self.static_vao = self._ui_vao(self._static_ui())
        azimuth, elevation = map(math.radians, (-50, 24))
        self.camera_direction = np.asarray(
            (
                math.cos(elevation) * math.cos(azimuth),
                math.sin(elevation),
                math.cos(elevation) * math.sin(azimuth),
            )
        )

    def _panel(self, size):
        accumulation = self.context.texture(size, 4, dtype="f2")
        weights = self.context.texture(size, 1, dtype="f2")
        resolved = self.context.texture(size, 4, dtype="f1")
        return (
            self.context.framebuffer((accumulation, weights)),
            self.context.framebuffer((resolved,)),
            accumulation,
            weights,
            resolved,
        )

    def _prepare_pair(self, first, second):
        pair = first, second
        if pair not in self.pair_vaos:
            frame0, frame1 = self.frames.data[first], self.frames.data[second]
            particles = self.context.vertex_array(
                self.program,
                (
                    (frame0, "4f 4f 2f", "p0", "s0", "q0"),
                    (frame1, "4f 4f 2f", "p1", "s1", "q1"),
                ),
            )
            arrows = self.context.vertex_array(
                self.arrow,
                (
                    (self.frames.arrow_data[first], "4f 4f", "p0", "s0"),
                    (self.frames.arrow_data[second], "4f 4f", "p1", "s1"),
                ),
            )
            self.pair_vaos[pair] = particles, arrows
        self.particle_vao, self.arrow_vao = self.pair_vaos[pair]

    def _matrices(self, lower, upper, amount):
        center = np.zeros(3)
        distance = self.frames.view_radius / math.sin(math.radians(19))
        perspective_matrix = perspective(
            38,
            1280 / 1080,
            distance - self.frames.view_radius,
            distance + self.frames.view_radius,
        ) @ look_at(distance * self.camera_direction, center, (0, 1, 0))
        radius = (1 - amount) * self.frames.detail_radii[lower] + amount * self.frames.detail_radii[upper]
        front = orthographic(radius, 640 / 360, 0.01, 4 * radius) @ look_at((0, 0, 2 * radius), center, (0, 1, 0))
        return perspective_matrix, front, radius

    def _render_panel(self, panel, amount, matrix, quantity, radius, surface):
        framebuffer, resolved, accumulation, weights, _ = panel
        framebuffer.use()
        framebuffer.clear(0, 0, 0, 0)
        self.context.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.context.blend_func = moderngl.ONE, moderngl.ONE
        self.program["mvp"].write(np.asarray(matrix.T, dtype="f4").tobytes())
        self.program["mix_value"].value = amount
        value_range = (self.frames.magnetic_range, self.frames.temperature_range, self.frames.density_range)[quantity]
        self.program["value_range"].value = tuple(map(math.log, value_range))
        self.program["point_scale"].value = framebuffer.height / (2 * math.tan(math.radians(19)) if surface else 2 * radius)
        self.program["view_radius"].value = radius
        self.program["slice_mode"].value = 0 if surface else 1
        self.program["quantity"].value = quantity
        self.program["surface_mode"].value = int(surface)
        self.particle_vao.render(moderngl.POINTS, vertices=self.frames.particle_count)
        self.context.disable(moderngl.BLEND)

        resolved.use()
        accumulation.use(0)
        weights.use(1)
        self.resolve["accumulated"].value = 0
        self.resolve["weights"].value = 1
        self.resolve["show_grid"].value = int(not surface)
        self.resolve["view_radius"].value = radius
        self.resolve["aspect"].value = 1280 / 1080 if surface else 640 / 360
        self.resolve["grid_spacing"].value = self.frames.view_radius / 8
        self.resolve_vao.render(vertices=6)

    def _draw_texture(self, texture, viewport, flip):
        self.final.viewport = viewport
        texture.use(0)
        self.blit["source"].value = 0
        self.blit["flip_y"].value = int(flip)
        self.blit_vao.render(vertices=6)

    def _draw_arrows(self, amount, radius):
        self.final.viewport = tuple(value * SCALE for value in (1280, 360, 640, 360))
        self.arrow["mix_value"].value = amount
        self.arrow["view_scale"].value = (radius * 640 / 360, radius)
        self.arrow["magnetic_range"].value = tuple(map(math.log, self.frames.magnetic_range))
        self.context.enable(moderngl.BLEND)
        self.context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        for colour, width in (((0, 0, 0, 0.88), 3.0), ((0.96, 0.98, 1, 0.96), 1.25)):
            self.arrow["arrow_colour"].value = colour
            self.context.line_width = width * SCALE
            self.arrow_vao.render(moderngl.POINTS, vertices=ARROW_COUNT)
        self.context.disable(moderngl.BLEND)

    def _ui_vao(self, vertices):
        array = np.asarray(vertices, dtype="f4")
        array[:, 0] = array[:, 0] / 960 - 1
        array[:, 1] = array[:, 1] / 540 - 1
        buffer = self.context.buffer(array.tobytes())
        return buffer, self.context.vertex_array(self.ui, [(buffer, "2f 4f", "position", "input_colour")])

    def _static_ui(self):
        colour = (0.82, 0.9, 1, 1)
        vertices = []
        labels = (
            (self.name, 30, 1050, 4),
            ("LOG DENSITY KG/M3", 35, 1010, 3),
            ("LOG TEMPERATURE K", 1310, 1050, 3),
            ("MAGNETIC FIELD T / DIRECTION X-Z", 1310, 700, 3),
            ("NORMALIZED HISTORY", 1310, 330, 3),
        )
        for label, x, y, pixel in labels:
            vertices.extend(text_vertices(label, x, y, pixel, colour))
        ranges = (
            (self.frames.density_range, 1240, 820),
            (self.frames.temperature_range, 1880, 760),
            (self.frames.magnetic_range, 1880, 400),
        )
        for limits, x, y in ranges:
            vertices.extend(text_vertices(f"{limits[1]:.1E}", x - 100, y + 180, 2, colour))
            vertices.extend(text_vertices(f"{limits[0]:.1E}", x - 100, y - 15, 2, colour))
            for index in range(48):
                vertices.extend(rectangle(x, y + 4 * index, x + 24, y + 4 * index + 4, (*turbo(index / 47), 1)))
        plot_left, plot_bottom, plot_width, plot_height = 1310, 45, 560, 220
        border = (0.28, 0.34, 0.42, 1)
        vertices.extend(line_quad((plot_left, plot_bottom), (plot_left + plot_width, plot_bottom), 1, border))
        vertices.extend(line_quad((plot_left, plot_bottom), (plot_left, plot_bottom + plot_height), 1, border))
        palette = ((0.2, 0.8, 1, 1), (1, 0.55, 0.2, 1), (1, 0.2, 0.5, 1), (0.5, 1, 0.4, 1))
        for index, symbol in enumerate("BTPV"):
            vertices.extend(text_vertices(symbol, 1670 + index * 55, 330, 3, palette[index]))
        x_values = plot_left + (self.frames.times - self.frames.times[0]) / (self.frames.times[-1] - self.frames.times[0]) * plot_width
        for x in x_values:
            vertices.extend(line_quad((x, plot_bottom), (x, plot_bottom - 7), 1, colour))
        start_label = f"{self.frames.times[0]:.1E}"
        end_label = f"{self.frames.times[-1]:.1E} S"
        vertices.extend(text_vertices(start_label, plot_left, 30, 2, colour))
        vertices.extend(
            text_vertices(
                end_label,
                plot_left + plot_width - len(end_label) * 12,
                30,
                2,
                colour,
            )
        )
        for column, line_colour in enumerate(palette):
            points = np.column_stack((x_values, plot_bottom + self.frames.statistics[:, column] * plot_height))
            for first, second in itertools.pairwise(points):
                vertices.extend(line_quad(first, second, 2, line_colour))
        separator = (0.12, 0.18, 0.25, 1)
        vertices.extend(line_quad((1280, 0), (1280, 1080), 2, separator))
        vertices.extend(line_quad((1280, 360), (1920, 360), 2, separator))
        vertices.extend(line_quad((1280, 720), (1920, 720), 2, separator))
        return vertices

    def _draw_ui(self, time, radius):
        colour = (0.82, 0.9, 1, 1)
        progress = (time - self.frames.times[0]) / (self.frames.times[-1] - self.frames.times[0])
        vertices = text_vertices(f"TIME {time:.3E} S", 35, 960, 3, colour)
        vertices.extend(text_vertices(f"R {radius:.2E} M", 1640, 1015, 2, colour))
        x = 1310 + progress * 560
        vertices.extend(line_quad((x, 45), (x, 265), 2, (1, 1, 1, 1)))
        buffer, vao = self._ui_vao(vertices)
        self.final.viewport = (0, 0, WIDTH, HEIGHT)
        self.static_vao.render(moderngl.TRIANGLES)
        vao.render(moderngl.TRIANGLES)
        vao.release()
        buffer.release()

    def render_into(self, time, target):
        upper = min(bisect.bisect_right(self.frames.times, time), len(self.frames.times) - 1)
        lower = max(0, upper - 1)
        amount = 0.0 if upper == lower else (time - self.frames.times[lower]) / (self.frames.times[upper] - self.frames.times[lower])
        self._prepare_pair(lower, upper)
        perspective_matrix, front_matrix, radius = self._matrices(lower, upper, amount)
        self._render_panel(
            self.main_panel,
            amount,
            perspective_matrix,
            2,
            self.frames.view_radius,
            surface=True,
        )
        self._render_panel(self.panels[0], amount, front_matrix, 1, radius, surface=False)
        self._render_panel(self.panels[1], amount, front_matrix, 0, radius, surface=False)
        self.final.use()
        self.final.clear(0.008, 0.012, 0.025, 1)
        self._draw_texture(self.main_panel[4], (0, 0, 2560, 2160), flip=False)
        self._draw_texture(self.panels[0][4], (2560, 1440, 1280, 720), flip=False)
        self._draw_texture(self.panels[1][4], (2560, 720, 1280, 720), flip=False)
        self._draw_arrows(amount, radius)
        self._draw_ui(time, radius)
        self.encoding.use()
        self.encoding.viewport = (0, 0, WIDTH, HEIGHT)
        self._draw_texture(self.final_texture, (0, 0, WIDTH, HEIGHT), flip=True)
        self.encoding.read_into(target, components=4, alignment=1)
