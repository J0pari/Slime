#!/usr/bin/env python3
"""
Visualization script for SLIME MNIST training.
Reads verified raw data and generates PDF report with emphasis on failures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
from pathlib import Path
from core import BinaryData
import struct

def collect_all_generations(report_dir):
    """Collect all generation data from raw files."""
    generations = []
    files = os.listdir(report_dir)

    gen_nums = set()
    for f in files:
        if '_' in f and '.raw' in f:
            parts = f.split('_')
            if len(parts) >= 2:
                try:
                    gen_num = int(parts[1].split('.')[0])
                    gen_nums.add(gen_num)
                except ValueError:
                    pass

    for gen in sorted(gen_nums):
        try:
            # Core data paths
            mnist_path = Path(f"{report_dir}/mnist_{gen:04d}.raw")
            ca_path = Path(f"{report_dir}/ca_{gen:04d}.raw")
            logits_path = Path(f"{report_dir}/logits_{gen:04d}.raw")
            label_path = Path(f"{report_dir}/label_{gen:04d}.raw")

            # Extended data paths
            rd_u_path = Path(f"{report_dir}/rd_u_{gen:04d}.raw")
            rd_v_path = Path(f"{report_dir}/rd_v_{gen:04d}.raw")
            resources_path = Path(f"{report_dir}/resources_{gen:04d}.raw")
            fitness_path = Path(f"{report_dir}/fitness_{gen:04d}.raw")
            lifecycle_path = Path(f"{report_dir}/lifecycle_{gen:04d}.raw")
            perf_path = Path(f"{report_dir}/perf_{gen:04d}.raw")

            if all(p.exists() for p in [mnist_path, ca_path, logits_path, label_path]):
                mnist = BinaryData.load_mnist(mnist_path)
                ca = BinaryData.load_ca_state(ca_path)
                logits = BinaryData.load_logits(logits_path)
                label = BinaryData.load_label(label_path)

                predicted = np.argmax(logits)
                loss = BinaryData.cross_entropy_loss(logits, label)
                correct = (predicted == label)

                gen_data = {
                    'gen': gen,
                    'mnist': mnist,
                    'ca': ca[:,:,0],
                    'logits': logits,
                    'label': label,
                    'predicted': predicted,
                    'loss': loss,
                    'correct': correct
                }

                # Load extended data if available
                if rd_u_path.exists():
                    gen_data['rd_u'] = BinaryData.load_field(rd_u_path)
                if rd_v_path.exists():
                    gen_data['rd_v'] = BinaryData.load_field(rd_v_path)
                if resources_path.exists():
                    gen_data['resources'] = BinaryData.load_field(resources_path)
                if fitness_path.exists():
                    gen_data['fitness'] = BinaryData.load_field(fitness_path)
                if lifecycle_path.exists():
                    gen_data['lifecycle'] = BinaryData.load_typed(lifecycle_path, np.int32)
                if perf_path.exists():
                    gen_data['perf'] = BinaryData.load_typed(perf_path, np.float32)

                generations.append(gen_data)
        except Exception as e:
            print(f"Warning: Failed to load generation {gen}: {e}")

    return generations

def generate_pdf_report(report_dir, output_pdf):
    """Generate comprehensive PDF report with failure emphasis."""
    print(f"Collecting data from {report_dir}...")
    generations = collect_all_generations(report_dir)

    if not generations:
        print("ERROR: No generation data found")
        return

    print(f"Found {len(generations)} generations")

    with PdfPages(output_pdf) as pdf:
        # Page 1: Training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        gens = [g['gen'] for g in generations]
        losses = [g['loss'] for g in generations]
        accuracies = [1.0 if g['correct'] else 0.0 for g in generations]

        axes[0, 0].plot(gens, losses, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Generation', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(gens, accuracies, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Generation', fontsize=12)
        axes[0, 1].set_ylabel('Correct (0 or 1)', fontsize=12)
        axes[0, 1].set_title('Per-Sample Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylim([-0.1, 1.1])
        axes[0, 1].grid(True, alpha=0.3)

        # Logit evolution
        for cls in range(10):
            logits_per_class = [g['logits'][cls] for g in generations]
            axes[1, 0].plot(gens, logits_per_class, label=f'Class {cls}', alpha=0.7)
        axes[1, 0].set_xlabel('Generation', fontsize=12)
        axes[1, 0].set_ylabel('Logit Value', fontsize=12)
        axes[1, 0].set_title('Logit Evolution', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=8, ncol=2)
        axes[1, 0].grid(True, alpha=0.3)

        # Confusion matrix
        confusion = np.zeros((10, 10), dtype=int)
        for g in generations:
            confusion[g['label'], g['predicted']] += 1

        sns.heatmap(confusion, annot=True, fmt='d', cmap='Reds', ax=axes[1, 1], cbar=True)
        axes[1, 1].set_xlabel('Predicted', fontsize=12)
        axes[1, 1].set_ylabel('True', fontsize=12)
        axes[1, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Page 2: FAILURES ONLY (most important!)
        failures = [g for g in generations if not g['correct']]
        print(f"Found {len(failures)} failures out of {len(generations)} total")

        if failures:
            n_show = min(12, len(failures))
            fig, axes = plt.subplots(3, 4, figsize=(14, 10))
            axes = axes.flatten()

            for i in range(n_show):
                g = failures[i]
                ax = axes[i]

                # Show MNIST | CA side by side
                combined = np.hstack([g['mnist'], g['ca'][:28, :28]])
                ax.imshow(combined, cmap='gray')
                ax.set_title(f"Gen {g['gen']}: Pred={g['predicted']} True={g['label']}\nLoss={g['loss']:.3f}",
                            color='red', fontsize=10, fontweight='bold')
                ax.axis('off')

            for i in range(n_show, 12):
                axes[i].axis('off')

            plt.suptitle(f'⚠ PREDICTION FAILURES ({len(failures)} total)',
                        fontsize=16, fontweight='bold', color='red')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        else:
            fig = plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, '✓ NO FAILURES\nAll predictions correct!',
                    ha='center', va='center', fontsize=24, color='green', fontweight='bold')
            plt.axis('off')
            pdf.savefig()
            plt.close()

        # Page 3: Successes (for comparison)
        successes = [g for g in generations if g['correct']]

        if successes:
            n_show = min(12, len(successes))
            fig, axes = plt.subplots(3, 4, figsize=(14, 10))
            axes = axes.flatten()

            for i in range(n_show):
                g = successes[i]
                ax = axes[i]

                combined = np.hstack([g['mnist'], g['ca'][:28, :28]])
                ax.imshow(combined, cmap='gray')
                ax.set_title(f"Gen {g['gen']}: Pred={g['predicted']} True={g['label']}\nLoss={g['loss']:.3f}",
                            color='green', fontsize=10)
                ax.axis('off')

            for i in range(n_show, 12):
                axes[i].axis('off')

            plt.suptitle(f'✓ CORRECT PREDICTIONS ({len(successes)} total)',
                        fontsize=16, fontweight='bold', color='green')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Page 4: Reaction-Diffusion Patterns
        rd_gens = [g for g in generations if 'rd_u' in g and 'rd_v' in g]
        if rd_gens:
            n_show = min(6, len(rd_gens))
            fig, axes = plt.subplots(3, 4, figsize=(14, 10))

            for i in range(n_show):
                g = rd_gens[i * len(rd_gens) // n_show]

                # U-field
                ax = axes[i // 2, (i % 2) * 2]
                im = ax.imshow(g['rd_u'], cmap='viridis', interpolation='bilinear')
                ax.set_title(f"Gen {g['gen']}: Activator U", fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

                # V-field
                ax = axes[i // 2, (i % 2) * 2 + 1]
                im = ax.imshow(g['rd_v'], cmap='plasma', interpolation='bilinear')
                ax.set_title(f"Gen {g['gen']}: Inhibitor V", fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

            # Hide unused subplots
            for i in range(n_show * 2, 12):
                axes.flatten()[i].axis('off')

            plt.suptitle('Gray-Scott Reaction-Diffusion Patterns', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Page 5: Resource Flow and Fitness Landscape
        flow_gens = [g for g in generations if 'resources' in g and 'fitness' in g]
        if flow_gens:
            n_show = min(6, len(flow_gens))
            fig, axes = plt.subplots(3, 4, figsize=(14, 10))

            for i in range(n_show):
                g = flow_gens[i * len(flow_gens) // n_show]

                # Resource density
                ax = axes[i // 2, (i % 2) * 2]
                im = ax.imshow(g['resources'], cmap='YlOrRd', interpolation='bilinear')
                ax.set_title(f"Gen {g['gen']}: Resource Density\nTotal={g['resources'].sum():.1f}", fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

                # Fitness landscape
                ax = axes[i // 2, (i % 2) * 2 + 1]
                im = ax.imshow(g['fitness'], cmap='coolwarm', interpolation='bilinear')
                ax.set_title(f"Gen {g['gen']}: Fitness Landscape\nMax={g['fitness'].max():.3f}", fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

            for i in range(n_show * 2, 12):
                axes.flatten()[i].axis('off')

            plt.suptitle('Resource Flow and Fitness Landscape', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Page 6: Lifecycle Phase Evolution
        lifecycle_gens = [g for g in generations if 'lifecycle' in g]
        if lifecycle_gens:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            phase_names = ['ACTIVE', 'STRESSED', 'DORMANT', 'REACTIVATING']
            colors = ['green', 'orange', 'gray', 'blue']

            # Stacked area chart
            gens_with_lc = [g['gen'] for g in lifecycle_gens]
            phase_data = np.array([g['lifecycle'] for g in lifecycle_gens]).T

            axes[0, 0].stackplot(gens_with_lc, phase_data, labels=phase_names, colors=colors, alpha=0.7)
            axes[0, 0].set_xlabel('Generation', fontsize=12)
            axes[0, 0].set_ylabel('Organism Count', fontsize=12)
            axes[0, 0].set_title('Lifecycle Phase Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].legend(loc='upper left', fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)

            # Individual phase trends
            for phase_idx, (name, color) in enumerate(zip(phase_names, colors)):
                axes[0, 1].plot(gens_with_lc, phase_data[phase_idx],
                               label=name, color=color, linewidth=2)
            axes[0, 1].set_xlabel('Generation', fontsize=12)
            axes[0, 1].set_ylabel('Count', fontsize=12)
            axes[0, 1].set_title('Phase Trends', fontsize=14, fontweight='bold')
            axes[0, 1].legend(fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)

            # Pie chart for final generation
            if lifecycle_gens:
                final_counts = lifecycle_gens[-1]['lifecycle']
                axes[1, 0].pie(final_counts, labels=phase_names, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title(f'Final Distribution (Gen {lifecycle_gens[-1]["gen"]})', fontsize=14, fontweight='bold')

            # Phase transition heatmap
            if len(lifecycle_gens) >= 2:
                normalized = phase_data / (phase_data.sum(axis=0, keepdims=True) + 1e-10)
                im = axes[1, 1].imshow(normalized, aspect='auto', cmap='YlGnBu', interpolation='nearest')
                axes[1, 1].set_xlabel('Generation Index', fontsize=12)
                axes[1, 1].set_ylabel('Phase', fontsize=12)
                axes[1, 1].set_yticks(range(4))
                axes[1, 1].set_yticklabels(phase_names)
                axes[1, 1].set_title('Normalized Phase Distribution', fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

            plt.suptitle('Lifecycle Dynamics', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"✓ Generated {output_pdf}")
    print(f"  Total samples: {len(generations)}")
    print(f"  Failures: {len(failures)} ({100*len(failures)/len(generations):.1f}%)")
    print(f"  Successes: {len(successes)} ({100*len(successes)/len(generations):.1f}%)")

def main():
    if len(sys.argv) < 2:
        print("Usage: visualize.py <report_directory> [output.pdf]")
        sys.exit(1)

    report_dir = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else "slime_mnist_report.pdf"

    if not os.path.exists(report_dir):
        print(f"ERROR: Directory not found: {report_dir}")
        sys.exit(1)

    generate_pdf_report(report_dir, output_pdf)

if __name__ == "__main__":
    main()
