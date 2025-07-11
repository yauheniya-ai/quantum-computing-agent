import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import List

# Set matplotlib backend before importing pyplot!
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

# 1. Define the quantum tool
@tool
def quantum_tool(task: str) -> str:
    """
    Runs a quantum circuit based on the task description.
    If the task includes 'hadamard', applies a Hadamard gate.
    Returns a summary and saves circuit, histogram, and Bloch sphere as images.
    """
    qc = QuantumCircuit(1, 1)
    if "hadamard" in task.lower():
        qc.h(0)
    qc.measure(0, 0)

    # Draw circuit and save as image
    fig = qc.draw(output='mpl')
    fig.savefig('circuit.png')

    # Run the circuit using the new Qiskit 1.0+ API
    simulator = Aer.get_backend('aer_simulator')
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts(compiled_circuit)

    # Plot histogram and save as image
    fig_hist = plot_histogram(counts)
    fig_hist.savefig('histogram.png')

    # Bloch sphere (before measurement)
    qc2 = QuantumCircuit(1)
    if "hadamard" in task.lower():
        qc2.h(0)
    qc2.save_statevector()
    compiled_circuit2 = transpile(qc2, simulator)
    result2 = simulator.run(compiled_circuit2).result()
    final_state = result2.get_statevector(compiled_circuit2)
    fig_bloch = plot_bloch_multivector(final_state)
    fig_bloch.savefig('bloch.png')

    return (
        f"Quantum result: {counts}\n"
        "Circuit diagram saved as circuit.png\n"
        "Histogram saved as histogram.png\n"
        "Bloch sphere saved as bloch.png"
    )

# 2. Set up the LLM and bind the tool
tools = [quantum_tool]
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).bind_tools(tools)

# 3. Define the agent node
def agent_node(state: MessagesState):
    messages: List = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

# 4. Conditional logic: decide if tool is needed
def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# 5. Build the LangGraph workflow
workflow = StateGraph(MessagesState)
tool_node = ToolNode(tools)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

# 6. Run the agent with a quantum prompt
messages = [HumanMessage(content="Please run a Hadamard gate and show me the quantum circuit.")]
result = graph.invoke({"messages": messages})

# Print the agent's reply (the quantum result is shown in the plots)
print(result["messages"][-1].content)
