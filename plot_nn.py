import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Layers
layers = [
    ("Input\n(3, 64, 64)", (0.05, 0.7), 'lightblue'),
    ("Conv1\n(64, 32, 32)", (0.25, 0.7), 'lightgreen'),
    ("Layer 1\n(64, 32, 32)", (0.45, 0.7), 'lightyellow'),
    ("Layer 2\n(128, 16, 16)", (0.45, 0.5), 'lightyellow'),
    ("Layer 3\n(256, 8, 8)", (0.45, 0.3), 'lightyellow'),
    ("Layer 4\n(512, 4, 4)", (0.45, 0.1), 'lightyellow'),
    ("AvgPool\n(512, 1, 1)", (0.65, 0.1), 'lightpink'),
    ("FC\n(1 logit)", (0.85, 0.1), 'salmon')
]

for name, (x, y), color in layers:
    rect = patches.Rectangle((x, y-0.05), 0.12, 0.1, linewidth=1.5, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + 0.06, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

# Arrows
arrows = [
    ((0.17, 0.7), (0.25, 0.7)),
    ((0.37, 0.7), (0.45, 0.7)),
    ((0.51, 0.65), (0.51, 0.6)),
    ((0.51, 0.45), (0.51, 0.4)),
    ((0.51, 0.25), (0.51, 0.2)),
    ((0.57, 0.1), (0.65, 0.1)),
    ((0.77, 0.1), (0.85, 0.1))
]

for (x1, y1), (x2, y2) in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2, color='gray'))

plt.title("ResNet18 Model Architecture Flow for Gravitational Lenses", fontsize=14, fontweight='bold', pad=20)
plt.savefig("neural_network_visual.png", dpi=300, bbox_inches='tight')
print("Successfully generated neural_network_visual.png")
