import sys
import re
import textwrap

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLineEdit, QPushButton, QTextEdit, QLabel, QProgressDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import networkx as nx
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Lazy-load transformers and torch to speed up initial GUI launch
llm_model = None
llm_tokenizer = None

class LLMThread(QThread):
    """Runs the LLM generation in a separate thread to avoid freezing the GUI."""
    generation_complete = pyqtSignal(str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        global llm_model, llm_tokenizer
        if llm_model is None or llm_tokenizer is None:
            try:
                from transformers import T5ForConditionalGeneration, T5Tokenizer
                model_name = 'google/flan-t5-base'
                llm_tokenizer = T5Tokenizer.from_pretrained(model_name)
                llm_model = T5ForConditionalGeneration.from_pretrained(model_name)
            except ImportError:
                self.generation_complete.emit("Error: transformers or torch is not installed.")
                return
            except Exception as e:
                self.generation_complete.emit(f"Error loading model: {e}")
                return

        try:
            input_ids = llm_tokenizer(self.prompt, return_tensors="pt").input_ids
            outputs = llm_model.generate(input_ids, max_length=512, temperature=0.1)
            result_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.generation_complete.emit(result_text)
        except Exception as e:
            self.generation_complete.emit(f"Error during generation: {e}")


class GraphCanvas(FigureCanvas):
    """A Matplotlib canvas to display the reasoning graph, integrated with PyQt."""
    node_clicked = pyqtSignal(str)

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.graph = None
        self.pos = None
        self.node_labels = {}
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def clear_graph(self):
        self.axes.clear()
        self.draw()

    def draw_graph(self, trace_steps):
        self.axes.clear()
        self.graph = nx.DiGraph()
        self.node_labels = {}

        if not trace_steps:
            self.axes.text(0.5, 0.5, 'No steps to display.', ha='center', va='center')
            self.draw()
            return

        for i, step in enumerate(trace_steps):
            node_id = i + 1
            self.graph.add_node(node_id, content=step)
            self.node_labels[node_id] = f"Step {node_id}"
            if i > 0:
                self.graph.add_edge(i, node_id)

        self.pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(
            self.graph, self.pos, ax=self.axes, with_labels=True,
            labels=self.node_labels,
            node_size=2000, node_color="skyblue", font_size=10, 
            font_weight="bold", arrows=True, arrowstyle='->',
            arrowsize=20
        )
        self.axes.set_title("LLM Reasoning Steps")
        self.draw()

    def on_click(self, event):
        if event.inaxes != self.axes or not self.pos:
            return

        x, y = event.xdata, event.ydata
        min_dist = float('inf')
        clicked_node = None

        for node, (nx, ny) in self.pos.items():
            dist = (nx - x)**2 + (ny - y)**2
            if dist < min_dist:
                min_dist = dist
                clicked_node = node
        
        # Heuristic to decide if click is close enough to a node
        if clicked_node is not None and min_dist < 0.01:
            node_content = self.graph.nodes[clicked_node]['content']
            self.node_clicked.emit(f"Step {clicked_node}:\n{node_content}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Reasoning Visualizer")
        self.setGeometry(100, 100, 1000, 700)

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel (Controls) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(350)

        # Input
        self.prompt_label = QLabel("Enter your question:")
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "e.g., A juggler has 16 balls. Half are red. He drops 4. How many red balls are left?"
        )
        self.prompt_input.setFixedHeight(150)

        # Button
        self.generate_button = QPushButton("Visualize Reasoning")
        self.generate_button.clicked.connect(self.start_generation)

        # Selected Node Display
        self.node_info_label = QLabel("Selected Step Details:")
        self.node_info_display = QTextEdit()
        self.node_info_display.setReadOnly(True)
        self.node_info_display.setFont(QFont("monospace", 10))
        self.node_info_display.setText("Click on a node in the graph to see its full text here.")

        left_layout.addWidget(self.prompt_label)
        left_layout.addWidget(self.prompt_input)
        left_layout.addWidget(self.generate_button)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.node_info_label)
        left_layout.addWidget(self.node_info_display)

        # --- Right Panel (Graph) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.graph_canvas = GraphCanvas(self)
        self.graph_canvas.node_clicked.connect(self.update_node_info)
        right_layout.addWidget(self.graph_canvas)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        self.llm_thread = None

    def start_generation(self):
        user_question = self.prompt_input.toPlainText().strip()
        if not user_question:
            self.update_node_info("Error: Please enter a question.")
            return

        prompt = (
            "You are a helpful assistant that thinks step-by-step. "
            "Solve the following problem. Format your entire response as a numbered list, where each line starts with 'Step X:'. "
            "Do not add any introductory or concluding sentences. Just provide the steps.\n\n" 
            f"Problem: {user_question}"
        )
        
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Generating... please wait")
        self.graph_canvas.clear_graph()
        self.update_node_info("Generating reasoning trace...")
        
        self.llm_thread = LLMThread(prompt)
        self.llm_thread.generation_complete.connect(self.process_llm_output)
        self.llm_thread.start()

    def process_llm_output(self, llm_output):
        self.generate_button.setEnabled(True)
        self.generate_button.setText("Visualize Reasoning")

        if llm_output.startswith("Error:"):
            self.update_node_info(llm_output)
            return

        steps = self.parse_trace(llm_output)
        if not steps:
            self.update_node_info(
                f"Could not parse steps from the model's output.\n\nRaw Output:\n{llm_output}"
            )
            self.graph_canvas.clear_graph()
        else:
            self.graph_canvas.draw_graph(steps)
            self.update_node_info("Graph generated. Click a node to see details.")

    def parse_trace(self, text):
        """Parses the LLM output into a list of reasoning steps."""
        # Regex to find lines starting with 'Step X:'
        pattern = re.compile(r"^Step\s*\d+:\s*(.*)", re.MULTILINE)
        matches = pattern.findall(text)
        # Clean up whitespace and ensure non-empty
        steps = [match.strip() for match in matches if match.strip()]
        return steps

    def update_node_info(self, text):
        wrapped_text = '\n'.join(textwrap.wrap(text, width=40))
        self.node_info_display.setText(text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
