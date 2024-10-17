from manim import *

class NeuralNetworkVisualization(Scene):
    def construct(self):
        # Input Layer
        input_values = [1, 0.5, 0.75, 0.9, 0.2, 0.4]  # Example input values
        input_nodes = self.create_nodes(input_values, LEFT)
        self.play(*[FadeIn(node) for node in input_nodes])
        self.wait(1)

        # Hidden Layer
        hidden_nodes = self.create_nodes([0, 0, 0], RIGHT)  # Placeholder for hidden layer values
        self.play(Transform(VGroup(*hidden_nodes), VGroup(*hidden_nodes)))
        self.wait(1)

        # Weights and Biases
        weights = [1.2, -0.5, 0.3]  # Example weights
        biases = [0.1, -0.2, 0.05]   # Example biases
        weight_lines = self.create_weight_lines(input_nodes, hidden_nodes, weights)
        self.play(*[Create(line) for line in weight_lines])
        self.wait(1)

        # Show Activation Function (Sigmoid)
        sigmoid_graph = self.plot_sigmoid_function()
        self.play(Create(sigmoid_graph))
        self.wait(2)

        # Output Layer
        output_node = Circle(radius=0.3, color=BLUE).shift(RIGHT * 3)
        self.play(FadeIn(output_node))
        self.wait(1)

        # Show final output (just a placeholder for visualization)
        output_text = Text("Output: 0.85", font_size=24).next_to(output_node, DOWN)
        self.play(FadeIn(output_text))
        self.wait(2)

    def create_nodes(self, values, position):
        nodes = []
        for i, value in enumerate(values):
            node = Circle(radius=0.3, color=GREEN).shift(position * (i + 1))
            label = Text(f"{value:.2f}").move_to(node.get_center())
            nodes.append(VGroup(node, label))
        return nodes

    def create_weight_lines(self, input_nodes, hidden_nodes, weights):
        lines = []
        for i, input_node in enumerate(input_nodes):
            for j, hidden_node in enumerate(hidden_nodes):
                line = Line(input_node.get_right(), hidden_node.get_left())
                line.set_color(YELLOW)
                lines.append(line)
        return lines

    def plot_sigmoid_function(self):
        sigmoid_graph = FunctionGraph(lambda x: 1 / (1 + np.exp(-x)), x_range=[-6, 6])
        sigmoid_graph.set_color(RED)
        return sigmoid_graph

# To render the scene:
# manim -pql visualize_ann.py NeuralNetworkVisualization

