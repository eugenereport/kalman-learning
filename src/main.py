import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from KalmanFilter import KalmanFilter


class KalmanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kalman Filter Application")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=1, fill="both")
        self.create_new_tab()

        # Групування кнопок у один рядок
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.add_tab_button = tk.Button(button_frame, text="Додати вкладку", command=self.create_new_tab)
        self.add_tab_button.grid(row=0, column=0, padx=5)

        self.clone_tab_button = tk.Button(button_frame, text="Клонувати вкладку", command=self.clone_current_tab)
        self.clone_tab_button.grid(row=0, column=1, padx=5)

        self.remove_tab_button = tk.Button(button_frame, text="Видалити поточну вкладку", command=self.remove_current_tab)
        self.remove_tab_button.grid(row=0, column=2, padx=5)

    def create_new_tab(self, parameters=None):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=f"Вкладка {len(self.notebook.tabs()) + 1}")

        fig, ax = plt.subplots(figsize=(5, 4))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_panel = tk.Frame(frame)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        frequency = tk.DoubleVar(value=parameters['frequency'] if parameters else 1.0)
        amplitude = tk.DoubleVar(value=parameters['amplitude'] if parameters else 5.0)
        offset = tk.DoubleVar(value=parameters['offset'] if parameters else 10.0)
        total_time = tk.DoubleVar(value=parameters['total_time'] if parameters else 1.0)
        Q = tk.DoubleVar(value=parameters['Q'] if parameters else 1.0)
        R = tk.DoubleVar(value=parameters['R'] if parameters else 10.0)
        P = tk.DoubleVar(value=parameters['P'] if parameters else 1.0)
        initial_state = tk.DoubleVar(value=parameters['initial_state'] if parameters else 0.0)

        self.add_parameter_controls(control_panel, "Частота:", frequency)
        self.add_parameter_controls(control_panel, "Амплітуда:", amplitude)
        self.add_parameter_controls(control_panel, "Зсув:", offset)
        self.add_parameter_controls(control_panel, "Загальний час:", total_time)
        self.add_parameter_controls(control_panel, "Q (Матриця коваріації шуму процесу):", Q)
        self.add_parameter_controls(control_panel, "R (Матриця коваріації шуму вимірювання):", R)
        self.add_parameter_controls(control_panel, "P (Початкова матриця коваріації):", P)
        self.add_parameter_controls(control_panel, "Початкова оцінка стану:", initial_state)

        tk.Button(control_panel, text="Перестроїти графік", command=lambda: self.redraw_graph(frequency, amplitude, offset, total_time, Q, R, P, initial_state, ax, canvas)).pack(pady=5)
        tk.Button(control_panel, text="Видалити поточний графік", command=lambda: self.clear_graph(ax, canvas)).pack(pady=5)

        self.redraw_graph(frequency, amplitude, offset, total_time, Q, R, P, initial_state, ax, canvas)

    def clone_current_tab(self):
        current_tab_index = self.notebook.index(self.notebook.select())
        if current_tab_index != -1:
            frame = self.notebook.nametowidget(self.notebook.tabs()[current_tab_index])
            control_panel = frame.winfo_children()[-1]
            parameters = {
                'frequency': control_panel.winfo_children()[1].get(),
                'amplitude': control_panel.winfo_children()[3].get(),
                'offset': control_panel.winfo_children()[5].get(),
                'total_time': control_panel.winfo_children()[7].get(),
                'Q': control_panel.winfo_children()[9].get(),
                'R': control_panel.winfo_children()[11].get(),
                'P': control_panel.winfo_children()[13].get(),
                'initial_state': control_panel.winfo_children()[15].get()
            }
            self.create_new_tab(parameters)

    def add_parameter_controls(self, frame, label_text, variable):
        tk.Label(frame, text=label_text).pack(anchor="w")
        tk.Entry(frame, textvariable=variable).pack(anchor="w", pady=2, fill=tk.X)

    def redraw_graph(self, frequency, amplitude, offset, total_time, Q, R, P, initial_state, ax, canvas):
        frequency = frequency.get()
        amplitude = amplitude.get()
        offset = offset.get()
        total_time = total_time.get()
        Q = np.array([[Q.get()]])
        R = np.array([[R.get()]])
        P = np.array([[P.get()]])
        initial_state = np.array([[initial_state.get()]])

        F = np.array([[1]])
        H = np.array([[1]])
        kf = KalmanFilter(F, H, Q, R, P, initial_state)

        sampling_interval = 0.001
        time_steps = np.arange(0, total_time, sampling_interval)
        true_signal = offset + amplitude * np.sin(2 * np.pi * frequency * time_steps)
        noisy_signal = [val + np.random.normal(0, np.sqrt(R[0][0])) for val in true_signal]

        kalman_estimates = []
        for measurement in noisy_signal:
            kf.predict()
            estimate = kf.update(measurement)
            kalman_estimates.append(estimate[0][0])

        variance_before = np.var(noisy_signal)
        variance_after = np.var(np.array(kalman_estimates) - true_signal)

        ax.clear()
        ax.plot(time_steps, noisy_signal, label='Шумовий сигнал', color='orange', linestyle='-', alpha=0.6)
        ax.plot(time_steps, true_signal, label='Справжній сигнал', linestyle='--', color='blue')
        ax.plot(time_steps, kalman_estimates, label='Оцінка фільтром Калмана', color='green')

        ax.set_xlabel('Час (с)')
        ax.set_ylabel('Значення')
        ax.set_title(f'Фільтр Калмана\nДисперсія до: {variance_before:.2f}, Після: {variance_after:.2f}')
        ax.legend()
        ax.grid()
        canvas.draw()

    def clear_graph(self, ax, canvas):
        ax.clear()
        canvas.draw()

    def remove_current_tab(self):
        current_tab_index = self.notebook.index(self.notebook.select())
        if current_tab_index != -1:
            self.notebook.forget(current_tab_index)

if __name__ == "__main__":
    root = tk.Tk()
    app = KalmanApp(root)
    root.mainloop()
