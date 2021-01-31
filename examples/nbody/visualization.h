#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

namespace vis
{
    struct Particle
    {
        glm::vec3 pos;
        glm::vec3 vel;
        float mass;
    };

    struct Window
    {
        Window(std::size_t particleCount) : particleCount(particleCount)
        {
            constexpr auto width = 1600;
            constexpr auto height = 1200;
            const auto title = "n-body " + std::to_string(particleCount);

            glfwSetErrorCallback(onError);

            if (!glfwInit())
                throw std::runtime_error("failed to init GLFW");

            // set window hints before creating window
            // list of available hints and their defaults: http://www.glfw.org/docs/3.0/window.html#window_hints
            glfwWindowHint(GLFW_DEPTH_BITS, 32);
            glfwWindowHint(GLFW_STENCIL_BITS, 0);
            glfwWindowHint(GLFW_FOCUSED, false);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            //#ifndef NDEBUG
            glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
            //#endif

            window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);

            glfwMakeContextCurrent(window);
            glfwSwapInterval(0); // vsync

            if (glewInit() != GLEW_OK)
                throw std::runtime_error("failed to init GLEW");

            GLuint vao;
            glGenVertexArrays(1, &vao);
            glBindVertexArray(vao);

            glGenBuffers(1, &vbo);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(Particle), nullptr, GL_DYNAMIC_DRAW);

            glVertexAttribPointer(
                0,
                3,
                GL_FLOAT,
                GL_FALSE,
                sizeof(Particle),
                reinterpret_cast<void*>(offsetof(Particle, pos)));
            glVertexAttribPointer(
                1,
                3,
                GL_FLOAT,
                GL_FALSE,
                sizeof(Particle),
                reinterpret_cast<void*>(offsetof(Particle, vel)));
            glVertexAttribPointer(
                2,
                1,
                GL_FLOAT,
                GL_FALSE,
                sizeof(Particle),
                reinterpret_cast<void*>(offsetof(Particle, mass)));

            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            glEnableVertexAttribArray(2);

            const auto vertexShader = glCreateShader(GL_VERTEX_SHADER);
            const auto fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
            glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
            for (auto shader : {vertexShader, fragmentShader})
            {
                glCompileShader(shader);
                GLint length;
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
                std::string log(length, '\0');
                glGetShaderInfoLog(shader, length, &length, log.data());
                std::cerr << "Shader info log:\n" << log << '\n';
            }

            const auto program = glCreateProgram();
            glAttachShader(program, vertexShader);
            glAttachShader(program, fragmentShader);

            glBindAttribLocation(program, 0, "pos");
            glBindAttribLocation(program, 1, "vel");
            glBindAttribLocation(program, 2, "mass");

            glLinkProgram(program);
            GLint length;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
            std::string log(length, '\0');
            glGetProgramInfoLog(program, length, &length, log.data());
            std::cerr << "Program info log:\n" << log << '\n';

            glUseProgram(program);

            const auto projection = glm::perspective(45.0f, (float) width / height, 0.1f, 10.0f);
            auto view = glm::translate(glm::mat4{1.0f}, {0, 0, -5.0f});
            view = glm::rotate(view, glm::radians(45.0f), {0, 1, 0});
            const auto m = projection * view;
            glUniformMatrix4fv(glGetUniformLocation(program, "matrix"), 1, GL_FALSE, glm::value_ptr(m));

            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

            glMatrixMode(GL_PROJECTION);
            glLoadMatrixf(glm::value_ptr(projection));
            glMatrixMode(GL_MODELVIEW);
            glLoadMatrixf(glm::value_ptr(view));
        }

        ~Window()
        {
            glDeleteBuffers(1, &vbo);

            glfwDestroyWindow(window);
            glfwTerminate();
        }

        void update(const Particle* particles)
        {
            glfwPollEvents();

            // particles[0].pos = glm::vec3{};

            glBufferSubData(GL_ARRAY_BUFFER, 0, particleCount * sizeof(Particle), particles);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glDrawArrays(GL_POINTS, 0, particleCount);

            glfwSwapBuffers(window);
        }

    private:
        static void onError(int error, const char* description)
        {
            std::cerr << "GLFW error " << error << ": " << description << '\n';
        }

        static constexpr auto vertexShaderSource = R"(
#version 330 core

uniform mat4 matrix;

in vec3 pos;
in vec3 vel;
in float mass;

out vec3 color;

void main() {
    color = clamp(abs(vel) / 10, 0.0, 1.0);
    gl_PointSize = min(mass / 10.0f, 10.0f);
    gl_Position = matrix * vec4(pos, 1.0);
}
)";

        static constexpr auto fragmentShaderSource = R"(
#version 330 core

in vec3 color;

void main() {
    gl_FragColor = vec4(color, 1.0);
}
)";

        std::size_t particleCount = 0;
        GLFWwindow* window = nullptr;
        GLuint vbo;
    };
} // namespace vis
