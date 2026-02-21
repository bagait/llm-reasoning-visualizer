# LLM Reasoning Process Visualizer

This tool parses complex reasoning traces from a Large Language Model (LLM), such as chain-of-thought, and generates an interactive graph visualization. It helps you understand, debug, and analyze the step-by-step logical flow of an AI's decision-making process.

![Screenshot of the LLM Reasoning Visualizer](./screenshot.png)
*A sample visualization of the LLM's reasoning for a simple logic puzzle.*

## Features

-   **Interactive Graph:** Visualizes the reasoning process as a directed graph, where each node is a distinct thought or step.
-   **Step-by-Step Analysis:** Click on any node in the graph to view the full text of that specific reasoning step.
-   **Local LLM Integration:** Uses a local `google/flan-t5-base` model via the `transformers` library, so no API keys are needed.
-   **User-Friendly GUI:** A simple interface built with PyQt6 to input your question and view the results.
-   **Asynchronous Processing:** The LLM runs in a background thread to keep the UI responsive, even during model loading and inference.

## Installation

1.  **Clone the repository:**
    bash
    git clone https://github.com/bagait/llm-reasoning-visualizer.git
    cd llm-reasoning-visualizer
    

2.  **Create a virtual environment (recommended):**
    bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    

3.  **Install the dependencies:**
    bash
    pip install -r requirements.txt
    
    *Note: `torch` can be a large download. If you have a CUDA-enabled GPU, you might want to install a GPU-specific version of PyTorch for better performance.* 

## Usage

Run the application from your terminal:

bash
python main.py


1.  The main window will appear.
2.  Enter a question or a problem that requires multi-step reasoning into the text box.
    *   Good examples involve logic, math, or planning (e.g., "If a snail climbs 5 feet up a 20-foot wall during the day and slides back 2 feet at night, how many days does it take to reach the top?").
3.  Click the **"Visualize Reasoning"** button.
4.  The first time you run it, the model will be downloaded, which may take a few minutes. A progress indicator will show in the terminal.
5.  Once the LLM generates its response, a graph will appear on the right.
6.  Click on any node (e.g., "Step 1") in the graph to see the detailed text for that step in the "Selected Step Details" panel on the left.

## How It Works

1.  **Prompt Engineering:** When you enter a question, the application wraps it in a carefully crafted prompt that instructs the LLM to break down its answer into a numbered list, with each line starting with `Step X:`.
2.  **LLM Inference:** The prompt is sent to a locally-run `google/flan-t5-base` model. The model generates a textual, step-by-step reasoning trace.
3.  **Trace Parsing:** A regular expression is used to parse the model's raw text output. It extracts each line beginning with `Step X:` and treats the content as a single reasoning step.
4.  **Graph Construction:** The `networkx` library is used to construct a directed graph. Each parsed step becomes a node, and edges are created sequentially to link `Step 1` to `Step 2`, `Step 2` to `Step 3`, and so on.
5.  **Visualization:** The `matplotlib` library, embedded within a `PyQt6` GUI, renders the `networkx` graph. The application listens for mouse clicks on the graph canvas to identify which node the user is interested in.
6.  **Interactivity:** When a node is clicked, its full content is displayed in the details panel, allowing for easy inspection of the model's logic at each stage.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
