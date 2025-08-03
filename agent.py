import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import List

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# Quantum imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector

# Load OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Predefined test circuits
PREDEFINED_CIRCUITS = {
    "hadamard": """
from qiskit import QuantumCircuit
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)
""",
    "x_gate": """
from qiskit import QuantumCircuit
qc = QuantumCircuit(1, 1)
qc.x(0)
qc.measure(0, 0)
""",
    "ry": """
from qiskit import QuantumCircuit
from numpy import pi
qc = QuantumCircuit(1, 1)
qc.ry(pi/4, 0)
qc.measure(0, 0)
""",
    "hh": """
from qiskit import QuantumCircuit
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.h(0)
qc.measure(0, 0)
""",
    "bell": """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
""",
    "cnot": """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.x(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
""",
    "swap": """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.swap(0, 1)
qc.measure([0, 1], [0, 1])
""",
    "entangled_ry": """
from qiskit import QuantumCircuit
from numpy import pi
qc = QuantumCircuit(2, 2)
qc.ry(pi/3, 0)
qc.ry(pi/4, 1)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1]),
""",
    "toffoli": """
from qiskit import QuantumCircuit
qc = QuantumCircuit(3, 3)
qc.ccx(0, 1, 2)
qc.measure([0, 1, 2], [0, 1, 2])
    """,
"fredkin": """
from qiskit import QuantumCircuit
qc = QuantumCircuit(3, 3)
qc.cswap(0, 1, 2)
qc.measure([0, 1, 2], [0, 1, 2])
"""
}

# 1. Define the quantum tool
@tool
def quantum_tool(task: str) -> str:
    """
    Runs a quantum circuit. Accepts either:
    - A keyword like 'hadamard', 'x_gate',  'ry', 'hh', 
    'bell', 'cnot', 'swap', 'entangled_ry', 'toffoli', 'fredkin'.
    - Raw Qiskit code (must define 'qc = QuantumCircuit(...)')
    """

    task_clean = task.strip().lower()

    # Match known keywords from natural language
    if task_clean in PREDEFINED_CIRCUITS:
        task = task_clean

    # Either use predefined or treat input as Qiskit code
    code = PREDEFINED_CIRCUITS.get(task, task)

    local_vars = {}
    try:
        exec(code, {}, local_vars)
        qc = local_vars.get("qc")
        if qc is None or not isinstance(qc, QuantumCircuit):
            return "Error: No valid QuantumCircuit named 'qc' found."
    except Exception as e:
        return f"Error in Qiskit code: {str(e)}"

    # Draw and save circuit diagram
    fig = qc.draw(output='mpl')
    fig.savefig('circuit.png')

    # Run simulation
    simulator = Aer.get_backend('aer_simulator')
    compiled = transpile(qc, simulator)
    job = simulator.run(compiled, shots=1024)
    counts = job.result().get_counts()

    fig_hist = plot_histogram(counts)
    for tick in fig_hist.axes[0].get_xticklabels():
        tick.set_rotation(0)
    fig_hist.savefig('histogram.png')

    # Attempt Bloch sphere (only if 1 qubit and no measurement)
    try:
        if qc.num_qubits == 1 and qc.num_clbits == 1:
            qc2 = QuantumCircuit(1)
            if "hadamard" in task_clean:
                qc2.h(0)
            elif "x_gate" in task_clean or "x" in task_clean:
                qc2.x(0)
            elif "ry" in task_clean:
                from numpy import pi
                qc2.ry(pi/4, 0)
            elif "hh" in task_clean:
                qc2.h(0)
                qc2.h(0)
            qc2.save_statevector()
            result2 = simulator.run(transpile(qc2, simulator)).result()
            bloch = plot_bloch_multivector(result2.get_statevector())
            bloch.savefig('bloch.png')
    except Exception as e:
        print(f"Skipping Bloch sphere: {e}")

    return (
        f"Quantum result: {counts}\n"
        "Circuit diagram saved as circuit.png\n"
        "Histogram saved as histogram.png\n"
        "Bloch sphere saved as bloch.png (if applicable)"
    )


# 2. Bind tool to LLM
tools = [quantum_tool]
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key
).bind_tools(tools)

# Add a system message to guide the behavior
system_messages = [
    SystemMessage(content=(
        "You are a quantum assistant. When a user requests to run or display a quantum circuit, "
        "you must call the tool `quantum_tool` with either a predefined circuit name (like 'hadamard', 'x_gate', 'hh', 'ry', "
        "'bell', 'cnot', 'swap', 'entangled_ry', 'toffoli', 'fredkin'), "
        "or provide raw Qiskit code. Don't reply directly if a tool call is needed."
    )),
]

# 3. Agent node
def agent_node(state: MessagesState):
    messages: List = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

# 4. Conditional edge: does the LLM want to use a tool?
def should_continue(state: MessagesState):
    last = state["messages"][-1]

    # Proceed to tool if tool_call exists
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # Stop if tool output is already in the chat
    if "Quantum result" in last.content:
        return END

    return END

# 5. Build LangGraph
workflow = StateGraph(MessagesState)
tool_node = ToolNode(tools)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

# 6. Run the system
human_message = HumanMessage(content="Please run a Hadamard gate and show me the quantum circuit.")
messages = system_messages + [human_message]
result = graph.invoke({"messages": messages})

print(result["messages"][-1].content)
