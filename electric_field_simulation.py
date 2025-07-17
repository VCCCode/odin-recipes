"""Electric Field Simulator

Cómo usar:
- Ejecuta este script para abrir una ventana interactiva.
- Click izquierdo: agrega una carga positiva.
- Shift + click izquierdo: agrega una carga negativa.
- Click derecho sobre una carga: la elimina.
- Presiona 'p' para activar/desactivar el modo sonda.
- Presiona 'm' para cambiar la magnitud de nuevas cargas.

Ideas para mejoras futuras al final del archivo.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np

K = 8.99e9  # Constante de Coulomb (N m^2/C^2)
DEFAULT_CHARGE = 1e-9  # Carga por defecto en Coulomb
GRID_RANGE = (-5, 5)  # Rango por defecto para la malla
GRID_DENSITY = 20  # Puntos por eje en la malla


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
        self.probe_annotation = self.ax.annotate(
            "", xy=(0, 1), xycoords="axes fraction", ha="left",
            va="top"
        )
        self.init_grid()
        self.update_plot()

    def init_grid(self):
        """Inicializa la malla de cálculo."""
        x = np.linspace(GRID_RANGE[0], GRID_RANGE[1], GRID_DENSITY)
        y = np.linspace(GRID_RANGE[0], GRID_RANGE[1], GRID_DENSITY)
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
        if dist[idx] < (GRID_RANGE[1] - GRID_RANGE[0]) / GRID_DENSITY:
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

    def update_plot(self):
        """Actualiza la figura con las cargas y líneas de campo."""
        self.ax.clear()
        Ex, Ey = self.compute_field()
        mag = np.sqrt(Ex**2 + Ey**2)
        self.ax.streamplot(self.X, self.Y, Ex, Ey, color=mag, cmap="inferno")
        if self.charges:
            positions = np.array([c.pos for c in self.charges])
            colors = [c.color for c in self.charges]
            self.ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=80, edgecolor="k")
        self.ax.set_xlim(GRID_RANGE)
        self.ax.set_ylim(GRID_RANGE)
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


def main():
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    sim = Simulation(ax)
    cid_click = fig.canvas.mpl_connect('button_press_event', sim.on_click)
    cid_motion = fig.canvas.mpl_connect('motion_notify_event', sim.on_motion)

    def on_key(event):
        if event.key == 'p':
            sim.toggle_probe()
        elif event.key == 'm':
            sim.ask_charge_value()

    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)


if __name__ == '__main__':
    main()

# Ideas para mejoras futuras
# - Añadir sliders para controlar la densidad de la malla y el rango.
# - Incluir visualización del potencial eléctrico.
# - Exportar configuraciones a archivo y cargar sesiones previas.
