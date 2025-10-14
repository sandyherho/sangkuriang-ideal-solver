"""
Professional 3D visualization for KdV solitons.

Creates animations with:
- 3D surface plots showing wave evolution
- Rotating viewpoint
- Conservation law tracking
- Professional styling
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
import matplotlib as mpl

# Professional styling
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 13
mpl.rcParams['axes.linewidth'] = 1.5


class Animator:
    """Professional KdV soliton animations."""
    
    @staticmethod
    def create_gif(result: dict, filename: str, output_dir: str = "outputs",
                  title: str = "KdV Soliton", fps: int = 30, dpi: int = 150,
                  view_3d: bool = True, colormap: str = 'plasma',
                  line_width: float = 2.5, alpha: float = 0.9):
        """
        Create animated GIF of KdV soliton evolution.
        
        Args:
            result: Results from solver.solve()
            filename: Output filename
            output_dir: Output directory
            title: Animation title
            fps: Frames per second
            dpi: Resolution
            view_3d: Use 3D surface plot (True) or 2D line (False)
            colormap: Matplotlib colormap
            line_width: Line width for 2D plots
            alpha: Transparency
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        x = result['x']
        t = result['t']
        u = result['u']
        mass = result['mass']
        energy = result['energy']
        params = result['params']
        
        mass_error = result['mass_error']
        energy_error = result['energy_error']
        velocity = params['soliton_velocity']
        
        print(f"    Creating animation ({len(t)} frames)...")
        
        if view_3d:
            fig, anim_obj = Animator._create_3d_animation(
                x, t, u, mass, energy, mass_error, energy_error,
                velocity, title, colormap
            )
        else:
            fig, anim_obj = Animator._create_2d_animation(
                x, t, u, mass, energy, mass_error, energy_error,
                velocity, title, colormap, line_width, alpha
            )
        
        print(f"    Saving GIF...")
        
        writer = animation.PillowWriter(fps=fps)
        
        with tqdm(total=len(t), desc="    Rendering", unit="frame") as pbar:
            def progress_callback(current_frame, total_frames):
                pbar.n = current_frame + 1
                pbar.refresh()
            
            anim_obj.save(filepath, writer=writer, dpi=dpi,
                         progress_callback=progress_callback)
        
        plt.close(fig)
        print(f"    ✓ Animation saved: {filepath}")
    
    @staticmethod
    def _create_3d_animation(x, t, u, mass, energy, mass_error, energy_error,
                            velocity, title, colormap):
        """Create 3D surface plot animation."""
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Styling
        ax.set_xlabel('Position [m]', fontsize=12, color='white', labelpad=12)
        ax.set_ylabel('Time [s]', fontsize=12, color='white', labelpad=12)
        ax.set_zlabel('Amplitude [m]', fontsize=12, color='white', labelpad=12)
        ax.tick_params(colors='white', which='both', labelsize=10)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.grid(True, alpha=0.2, color='white', linestyle='--', linewidth=0.5)
        
        cmap = plt.get_cmap(colormap)
        
        # Create dummy surface for colorbar
        dummy_surf = ax.plot_surface(
            np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)),
            cmap=colormap, vmin=np.min(u), vmax=np.max(u)
        )
        cbar = fig.colorbar(dummy_surf, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label('Amplitude [m]', color='white', fontsize=18)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        dummy_surf.remove()
        
        title_text = ax.text2D(
            0.5, 0.95, '', transform=ax.transAxes,
            fontsize=14, color='white', weight='bold',
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='black',
                     alpha=0.7, edgecolor='white', linewidth=1.5)
        )
        
        stats_text = ax.text2D(
            0.5, 0.02, '', transform=ax.transAxes,
            fontsize=10, color='white', ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='black',
                     alpha=0.7, edgecolor='cyan', linewidth=1.5)
        )
        
        def animate(frame):
            ax.clear()
            ax.set_facecolor('black')
            
            # Surface up to current time
            u_subset = u[:frame+1].T
            t_subset = t[:frame+1]
            T_sub, X_sub = np.meshgrid(t_subset, x)
            
            surf = ax.plot_surface(
                X_sub, T_sub, u_subset,
                cmap=colormap, alpha=0.85,
                linewidth=0, antialiased=True,
                vmin=np.min(u), vmax=np.max(u)
            )
            
            # Current profile highlight
            ax.plot(x, [t[frame]]*len(x), u[frame],
                   color='cyan', linewidth=3.5, alpha=1.0, zorder=10)
            
            # REMOVED: Red soliton position marker
            
            ax.set_xlabel('Position [m]', fontsize=16, color='white', labelpad=12)
            ax.set_ylabel('Time [s]', fontsize=16, color='white', labelpad=12)
            ax.set_zlabel('Amplitude [m]', fontsize=16, color='white', labelpad=12)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(0, t[-1])
            ax.set_zlim(np.min(u)*1.1, np.max(u)*1.15)
            
            # Rotating view
            ax.view_init(elev=25, azim=45 + frame*0.2)
            
            ax.tick_params(colors='white', which='both', labelsize=14)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('white')
            ax.yaxis.pane.set_edgecolor('white')
            ax.zaxis.pane.set_edgecolor('white')
            ax.grid(True, alpha=0.2, color='white', linestyle='--', linewidth=0.5)
            
            # Calculate expected position for display only (not plotted)
            amplitude = np.max(u[0])
            expected_pos = np.mean(x[u[0] > 0.5*amplitude]) + velocity * t[frame]
            
            title_text.set_text(
                f'{title} | t = {t[frame]:.2f}s | Position: {expected_pos:.2f}m'
            )
            
            stats_text.set_text(
                f'Mass Δ: {mass_error:.2e} | Energy Δ: {energy_error:.2e} | '
                f'v = {velocity:.3f} m/s'
            )
            
            return [surf, title_text, stats_text]
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(t),
            interval=1000/30, blit=False, repeat=True
        )
        
        return fig, anim
    
    @staticmethod
    def _create_2d_animation(x, t, u, mass, energy, mass_error, energy_error,
                            velocity, title, colormap, line_width, alpha):
        """Create 2D line plot animation."""
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        ax.set_xlim(x[0], x[-1])
        y_margin = 0.1 * (np.max(u) - np.min(u))
        ax.set_ylim(np.min(u) - y_margin, np.max(u) + y_margin)
        ax.set_xlabel('Position [m]', fontsize=16, color='white')
        ax.set_ylabel('Amplitude [m]', fontsize=16, color='white')
        ax.tick_params(colors='white', which='both')
        
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.grid(True, alpha=0.25, color='white', linestyle='--', linewidth=0.5)
        
        cmap = plt.get_cmap(colormap)
        line, = ax.plot([], [], linewidth=line_width, color=cmap(0.7), alpha=alpha)
        line_glow, = ax.plot([], [], linewidth=line_width*2.5,
                           color=cmap(0.7), alpha=alpha*0.25)
        
        # REMOVED: Red marker initialization
        
        time_text = ax.text(
            0.02, 0.95, '', transform=ax.transAxes,
            fontsize=18, color='white', verticalalignment='top',
            weight='bold',
            bbox=dict(boxstyle='round', facecolor='black',
                     alpha=0.6, edgecolor=cmap(0.7), linewidth=2)
        )
        
        stats_text = ax.text(
            0.98, 0.95, '', transform=ax.transAxes,
            fontsize=14, color='white',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='black',
                     alpha=0.6, edgecolor='cyan', linewidth=2)
        )
        
        def animate(frame):
            color_val = frame / len(t)
            current_color = cmap(0.3 + 0.6 * color_val)
            
            line.set_data(x, u[frame])
            line.set_color(current_color)
            line_glow.set_data(x, u[frame])
            line_glow.set_color(current_color)
            
            # REMOVED: Red marker update
            
            time_text.set_text(f'{title}\nt = {t[frame]:.2f}s')
            time_text.get_bbox_patch().set_edgecolor(current_color)
            
            stats_text.set_text(
                f'Mass Δ: {mass_error:.2e}\n'
                f'Energy Δ: {energy_error:.2e}\n'
                f'v = {velocity:.3f} m/s'
            )
            
            return [line, line_glow, time_text, stats_text]
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(t),
            interval=1000/30, blit=False, repeat=True
        )
        
        return fig, anim
