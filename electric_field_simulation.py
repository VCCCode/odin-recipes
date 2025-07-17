"""Electric Field Simulator

Cómo usar:
- Ejecuta este script para abrir una ventana interactiva.
- Click izquierdo: agrega una carga positiva.
- Shift + click izquierdo: agrega una carga negativa.
- Click derecho sobre una carga: la elimina.
- Presiona 'p' para activar/desactivar el modo sonda.
- Presiona 'm' para cambiar la magnitud de nuevas cargas.
- Utiliza los sliders para ajustar la densidad de la malla y el rango.
- Los botones *Guardar* y *Cargar* permiten almacenar la sesión.

Ideas para mejoras futuras al final del archivo.
"""

import json
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import TextBox
import numpy as np

K = 8.99e9  # Constante de Coulomb (N m^2/C^2)
DEFAULT_CHARGE = 1e-9  # Carga por defecto en Coulomb
DEFAULT_RANGE = 5  # Extremo de los ejes x e y por defecto
DEFAULT_DENSITY = 20  # Puntos por eje en la malla


class Charge:
    """Representa una carga puntual en 2-D."""

    def __init__(self, q, pos):
        self.q = q
        self.pos = np.asarray(pos, dtype=float)
        self.color = "red" if q > 0 else "blue"

    def field_at(self, point):
        """Devuelve el campo eléctrico en un punto.

        Parameters
        ----------
        point : array_like
            Coordenada (x, y) donde evaluar el campo.

        Returns
        -------
        ndarray
            Vector campo eléctrico en el punto.
        """
        point = np.asarray(point, dtype=float)
        r = point - self.pos
        dist_sq = np.sum(r ** 2, axis=-1)
        # Evitar división por cero
        mask = dist_sq == 0
        dist_sq = np.where(mask, np.inf, dist_sq)
        r_norm = np.sqrt(dist_sq)
        field = K * self.q * r / (dist_sq * r_norm)[..., None]
        field[mask] = 0.0
        return field


class Simulation:
    """Gestiona la simulación de cargas."""

    def __init__(self, ax):
        self.ax = ax
        self.charges = []
        self.charge_value = DEFAULT_CHARGE
        self.probe_mode = False
        self.text_box = None
        self.cbar = None
        self.range_val = DEFAULT_RANGE
        self.grid_range = (-self.range_val, self.range_val)
        self.grid_density = DEFAULT_DENSITY
        self.probe_annotation = self.ax.annotate(
            "", xy=(0, 1), xycoords="axes fraction", ha="left",
            va="top"
        )
        self.init_grid()
        self.update_plot()

    def init_grid(self):
        """Inicializa la malla de cálculo."""
        x = np.linspace(self.grid_range[0], self.grid_range[1], self.grid_density)
        y = np.linspace(self.grid_range[0], self.grid_range[1], self.grid_density)
        self.X, self.Y = np.meshgrid(x, y)

    def add_charge(self, pos, q=None):
        """Agrega una carga en `pos`."""
        q = self.charge_value if q is None else q
        self.charges.append(Charge(q, pos))
        self.update_plot()

    def remove_charge(self, pos):
        """Elimina la carga más cercana a `pos` si está lo suficientemente cerca."""
        if not self.charges:
            return
        pos = np.asarray(pos)
        positions = np.array([c.pos for c in self.charges])
        dist = np.linalg.norm(positions - pos, axis=1)
        idx = np.argmin(dist)
        if dist[idx] < (self.grid_range[1] - self.grid_range[0]) / self.grid_density:
            self.charges.pop(idx)
            self.update_plot()

    def compute_field(self):
        """Calcula el campo resultante en la malla."""
        Ex = np.zeros_like(self.X)
        Ey = np.zeros_like(self.Y)
        for c in self.charges:
            field = c.field_at(np.stack([self.X, self.Y], axis=-1))
            Ex += field[..., 0]
            Ey += field[..., 1]
        return Ex, Ey

    def compute_potential(self):
        """Calcula el potencial eléctrico en la malla."""
        V = np.zeros_like(self.X)
        for c in self.charges:
            dx = self.X - c.pos[0]
            dy = self.Y - c.pos[1]
            r = np.hypot(dx, dy)
            r = np.where(r == 0, np.inf, r)
            V += K * c.q / r
        return V

    def update_plot(self):
        """Actualiza la figura con las cargas y líneas de campo."""
        self.ax.clear()
        Ex, Ey = self.compute_field()
        V = self.compute_potential()
        mag = np.sqrt(Ex**2 + Ey**2)
        im = self.ax.imshow(
            V,
            extent=[self.grid_range[0], self.grid_range[1], self.grid_range[0], self.grid_range[1]],
            origin="lower",
            cmap="viridis",
            alpha=0.6,
        )
        if self.cbar:
            self.cbar.remove()
        self.cbar = self.ax.figure.colorbar(im, ax=self.ax)
        self.ax.streamplot(self.X, self.Y, Ex, Ey, color=mag, cmap="inferno")
        if self.charges:
            positions = np.array([c.pos for c in self.charges])
            colors = [c.color for c in self.charges]
            self.ax.scatter(
                positions[:, 0], positions[:, 1], c=colors, s=80, edgecolor="k"
            )
        self.ax.set_xlim(self.grid_range)
        self.ax.set_ylim(self.grid_range)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_title("Simulación de Campo Eléctrico")
        self.probe_annotation.set_text("")
        self.ax.figure.canvas.draw_idle()

    def on_click(self, event):
        """Maneja los clics del mouse."""
        if event.inaxes != self.ax:
            return
        pos = (event.xdata, event.ydata)
        if event.button == 1:  # izquierdo
            q = self.charge_value
            if event.key == "shift":
                q = -abs(q)
            self.add_charge(pos, q)
        elif event.button == 3:  # derecho
            self.remove_charge(pos)

    def toggle_probe(self):
        self.probe_mode = not self.probe_mode
        if not self.probe_mode:
            self.probe_annotation.set_text("")
            self.ax.figure.canvas.draw_idle()

    def on_motion(self, event):
        """Actualiza valores de la sonda mientras se mueve el cursor."""
        if not self.probe_mode or event.inaxes != self.ax:
            return
        point = np.array([event.xdata, event.ydata])
        E = np.sum([c.field_at(point) for c in self.charges], axis=0)
        E_mag = np.linalg.norm(E)
        F = E * self.charge_value
        text = f"|E|={E_mag:.2e} N/C\n|F|={np.linalg.norm(F):.2e} N"
        self.probe_annotation.set_text(text)
        self.ax.figure.canvas.draw_idle()

    def ask_charge_value(self):
        """Muestra un TextBox para definir la magnitud de las nuevas cargas."""
        if self.text_box:
            return  # Ya existe
        axbox = self.ax.inset_axes([0.7, 0.9, 0.25, 0.05])
        self.text_box = TextBox(axbox, 'q [C]', initial=f'{self.charge_value:.2e}')

        def submit(text):
            try:
                self.charge_value = float(text)
            except ValueError:
                pass
            axbox.remove()
            self.text_box = None
            self.ax.figure.canvas.draw_idle()

        self.text_box.on_submit(submit)
        self.ax.figure.canvas.draw_idle()

    def save_state(self):
        """Guarda la configuración actual en un archivo JSON."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if not filename:
            return
        state = {
            "charges": [
                {"q": c.q, "pos": c.pos.tolist()} for c in self.charges
            ],
            "density": self.grid_density,
            "range": self.range_val,
            "charge_value": self.charge_value,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Carga una configuración previamente guardada."""
        filename = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not filename:
            return
        with open(filename, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.charges = [Charge(c["q"], c["pos"]) for c in state.get("charges", [])]
        self.grid_density = int(state.get("density", self.grid_density))
        self.range_val = float(state.get("range", self.range_val))
        self.charge_value = state.get("charge_value", self.charge_value)
        self.grid_range = (-self.range_val, self.range_val)
        self.init_grid()
        self.update_plot()


def main():
    root = tk.Tk()
    root.title("Campo Eléctrico 2-D")

    fig, ax = plt.subplots(figsize=(6, 6))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    sim = Simulation(ax)

    fig.canvas.mpl_connect("button_press_event", sim.on_click)
    fig.canvas.mpl_connect("motion_notify_event", sim.on_motion)

    def on_key(event):
        if event.key == "p":
            sim.toggle_probe()
        elif event.key == "m":
            sim.ask_charge_value()

    fig.canvas.mpl_connect("key_press_event", on_key)

    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.RIGHT, fill=tk.Y)

    def update_density(val):
        sim.grid_density = int(float(val))
        sim.init_grid()
        sim.update_plot()

    density = tk.Scale(
        control_frame,
        from_=10,
        to=100,
        label="Densidad de malla",
        orient=tk.HORIZONTAL,
        command=update_density,
    )
    density.set(DEFAULT_DENSITY)
    density.pack(fill=tk.X, padx=5, pady=5)

    def update_range(val):
        sim.range_val = float(val)
        sim.grid_range = (-sim.range_val, sim.range_val)
        sim.init_grid()
        sim.update_plot()

    rango = tk.Scale(
        control_frame,
        from_=5,
        to=50,
        label="Rango",
        orient=tk.HORIZONTAL,
        command=update_range,
    )
    rango.set(DEFAULT_RANGE)
    rango.pack(fill=tk.X, padx=5, pady=5)

    tk.Button(control_frame, text="Guardar", command=sim.save_state).pack(fill=tk.X, padx=5, pady=5)
    tk.Button(control_frame, text="Cargar", command=sim.load_state).pack(fill=tk.X, padx=5, pady=5)

    root.mainloop()


if __name__ == '__main__':
    main()

# Ideas para mejoras futuras
# - Implementar soporte para movimiento de cargas en el tiempo.
# - Añadir visualizaciones 3-D.
# - Permitir definir condiciones de frontera y conductores.
