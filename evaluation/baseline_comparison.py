"""
Baseline Comparison Methods for PINN Evaluation.

This module implements baseline methods for comparing PINN performance:
1. Equal Area Criterion (EAC) - Analytical method
2. Traditional Time-Domain Simulation (TDS) - ANDES simulation
3. Standard ML Approaches - LSTM, CNN, etc. without physics
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False
    print("Warning: ANDES not available. TDS baseline will not work.")

from examples.scripts.utils.eac_calculator import calculate_cct_eac
from utils.metrics import compute_cct_metrics, compute_parameter_metrics, compute_trajectory_metrics


class EACBaseline:
    """
    Equal Area Criterion baseline for CCT estimation.

    This is an analytical method that provides approximate CCT values
    for simplified SMIB systems (GENCLS model, no damping)
    """

    def __init__(self):
        """Initialize EAC baseline."""
        self.name = "EAC (Equal Area Criterion)"
        self.method_type = "analytical"

    def estimate_cct(
        self,
        Pm: float,
        M: float,
        D: float,
        Xprefault: float,
        Xfault: float,
        Xpostfault: float,
        V1: float = 1.05,
        V2: float = 1.0,
    ) -> Optional[float]:
        """
        Estimate CCT using Equal Area Criterion.

        Parameters:
        -----------
        Pm : float
            Mechanical power (pu)
        M : float
            Inertia constant (seconds)
        D : float
            Damping coefficient (pu) - Note: EAC ignores damping
        Xprefault : float
            Pre-fault reactance (pu)
        Xfault : float
            Fault reactance (pu)
        Xpostfault : float
            Post-fault reactance (pu)
        V1 : float
            Generator voltage (pu)
        V2 : float
            Infinite bus voltage (pu)

        Returns:
        --------
        cct : float or None
            Estimated CCT (seconds), or None if calculation fails
        """
        return calculate_cct_eac(
            Pm=Pm, M=M, D=D, Xprefault=Xprefault, Xfault=Xfault, Xpostfault=Xpostfault, V1=V1, V2=V2
        )

    def predict_trajectory(
        self,
        t: np.ndarray,
        delta0: float,
        omega0: float,
        Pm: float,
        M: float,
        D: float,
        Xprefault: float,
        Xfault: float,
        Xpostfault: float,
        tf: float,
        tc: float,
        V1: float = 1.05,
        V2: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trajectory using simplified analytical solution.

        Note: EAC doesn't directly provide trajectories, so this uses
        a simplified numerical integration of the swing equation.

        Parameters:
        -----------
        t : np.ndarray
            Time points
        delta0 : float
            Initial rotor angle (rad)
        omega0 : float
            Initial rotor speed (pu)
        Pm : float
            Mechanical power (pu)
        M : float
            Inertia constant (seconds)
        D : float
            Damping coefficient (pu)
        Xprefault : float
            Pre-fault reactance (pu)
        Xfault : float
            Fault reactance (pu)
        Xpostfault : float
            Post-fault reactance (pu)
        tf : float
            Fault start time (seconds)
        tc : float
            Fault clear time (seconds)
        V1 : float
            Generator voltage (pu)
        V2 : float
            Infinite bus voltage (pu)

        Returns:
        --------
        tuple : (delta, omega)
            Predicted trajectories
        """
        # Simple numerical integration (Euler method).
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        delta = np.zeros_like(t)
        omega = np.zeros_like(t)

        delta[0] = delta0
        omega[0] = omega0

        for i in range(1, len(t)):
            # Determine system state
            if t[i] < tf:
                X = Xprefault
            elif t[i] <= tc:
                X = Xfault
            else:
                X = Xpostfault

            # Compute electrical power
            Pe = (V1 * V2 / X) * np.sin(delta[i - 1])

            # Swing equation: M·d²δ/dt² = Pm - Pe - D·dδ/dt
            # d²δ/dt² = (Pm - Pe - D·(ω-1)) / M
            domega_dt = (Pm - Pe - D * (omega[i - 1] - 1.0)) / M
            ddelta_dt = 2 * np.pi * 60 * (omega[i - 1] - 1.0)  # Convert to rad/s

            # Update
            omega[i] = omega[i - 1] + domega_dt * dt
            delta[i] = delta[i - 1] + ddelta_dt * dt

        return delta, omega


class TDSBaseline:
    """
    Traditional Time-Domain Simulation baseline using ANDES.

    This provides ground truth trajectories and CCT values by running
    actual time-domain simulations.
    """

    def __init__(self, case_file: str = "smib/SMIB.json"):
        """
        Initialize TDS baseline.

        Parameters:
        -----------
        case_file : str
            ANDES case file path
        """
        self.name = "TDS (Time-Domain Simulation)"
        self.method_type = "simulation"
        self.case_file = case_file
        self.andes_available = ANDES_AVAILABLE

    def simulate_trajectory(
        self,
        Pm: float,
        M: float,
        D: float,
        tf: float,
        tc: float,
        t_end: float = 5.0,
        time_step: Optional[float] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Simulate trajectory using ANDES TDS.

        Parameters:
        -----------
        Pm : float
            Mechanical power (pu)
        M : float
            Inertia constant (seconds)
        D : float
            Damping coefficient (pu)
        tf : float
            Fault start time (seconds)
        tc : float
            Fault clear time (seconds)
        t_end : float
            Simulation end time (seconds)
        time_step : float, optional
            Time step (seconds). If None, uses ANDES default.

        Returns:
        --------
        results : dict or None
            Dictionary with 'time', 'delta', 'omega', or None if simulation fails
        """
        if not self.andes_available:
            return None

        try:
            # Load system
            case_path = andes.get_case(self.case_file)
            ss = andes.load(case_path, default_config=True, no_output=True)

            # Set generator parameters
            if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
                ss.GENCLS.tm0.v[0] = Pm
                ss.GENCLS.M.v[0] = M
                ss.GENCLS.D.v[0] = D

            # Configure fault
            if hasattr(ss, "Fault") and ss.Fault.n > 0:
                ss.Fault.t1.v[0] = tf
                ss.Fault.t2.v[0] = tc

            # Run power flow
            ss.PFlow.run()

            # Configure TDS
            if time_step is not None:
                ss.TDS.config.h = time_step
            ss.TDS.config.criteria = 0  # Disable early stopping

            # Run TDS
            ss.TDS.run(tf=t_end)

            # Extract results
            if hasattr(ss.TDS, "plt") and hasattr(ss.TDS.plt, "GENCLS"):
                time_data = ss.TDS.plt.t
                delta_data = ss.TDS.plt.GENCLS.delta[:, 0]
                omega_data = ss.TDS.plt.GENCLS.omega[:, 0]

                return {"time": time_data, "delta": delta_data, "omega": omega_data}

            return None

        except Exception as e:
            print(f"TDS simulation failed: {e}")
            return None

    def find_cct(
        self,
        Pm: float,
        M: float,
        D: float,
        tf: float = 1.0,
        cct_min: float = 0.1,
        cct_max: float = 0.5,
        tolerance: float = 0.001,
        max_iter: int = 20,
    ) -> Optional[float]:
        """
        Find CCT using binary search with TDS.

        Parameters:
        -----------
        Pm : float
            Mechanical power (pu)
        M : float
            Inertia constant (seconds)
        D : float
            Damping coefficient (pu)
        tf : float
            Fault start time (seconds)
        cct_min : float
            Minimum CCT guess (seconds)
        cct_max : float
            Maximum CCT guess (seconds)
        tolerance : float
            Convergence tolerance (seconds)
        max_iter : int
            Maximum iterations

        Returns:
        --------
        cct : float or None
            Estimated CCT (seconds), or None if search fails
        """
        if not self.andes_available:
            return None

        # Binary search
        low, high = cct_min, cct_max

        for _ in range(max_iter):
            if high - low < tolerance:
                return (low + high) / 2

            tc = (low + high) / 2

            # Simulate
            results = self.simulate_trajectory(Pm, M, D, tf, tc)

            if results is None:
                return None

            # Check stability (simple: max angle < 180 degrees)
            delta_max = np.max(np.abs(results["delta"]))
            is_stable = delta_max < np.pi  # 180 degrees in radians

            if is_stable:
                low = tc
            else:
                high = tc

        return (low + high) / 2


class MLBaseline:
    """
    Standard Machine Learning baseline (LSTM, CNN, etc.) without physics.

    This provides a comparison with traditional ML approaches that don't
    incorporate physics constraints.
    """

    def __init__(self, model_type: str = "LSTM", **model_kwargs):
        """
        Initialize ML baseline.

        Parameters:
        -----------
        model_type : str
            Model type: "LSTM", "CNN", "Transformer"
        **model_kwargs
            Additional model parameters
        """
        self.name = "ML ({model_type})"
        self.method_type = "machine_learning"
        self.model_type = model_type
        self.model = None
        self.model_kwargs = model_kwargs

    def build_model(self, input_dim: int, output_dim: int = 2):
        """Build the ML model."""
        if self.model_type == "LSTM":
            self.model = self._build_lstm(input_dim, output_dim)
        elif self.model_type == "CNN":
            self.model = self._build_cnn(input_dim, output_dim)
        else:
            raise ValueError("Unknown model type: {self.model_type}")

    def _build_lstm(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build LSTM model."""
        hidden_size = self.model_kwargs.get("hidden_size", 64)
        num_layers = self.model_kwargs.get("num_layers", 2)

        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)
                # Take last output
                output = self.fc(lstm_out[:, -1, :])
                return output

        return LSTMModel()

    def _build_cnn(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build CNN model."""

        class CNNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
                self.fc = nn.Linear(32, output_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, seq_len, features)
                x = x.transpose(1, 2)  # (batch, features, seq_len)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = x.mean(dim=2)  # Global average pooling
                return self.fc(x)

        return CNNModel()

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, batch_size: int = 32
    ) -> None:
        """Train the ML model."""
        if self.model is None:
            self.build_model(X_train.shape[-1], y_train.shape[-1])

        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = torch.FloatTensor(X_train[i : i + batch_size])
                batch_y = torch.FloatTensor(y_train[i : i + batch_size])

                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            pred = self.model(X_tensor)
            return pred.numpy()


class BaselineComparator:
    """
    Comprehensive baseline comparison framework.

    Compares PINN performance against multiple baseline methods:
    - EAC (analytical)
    - TDS (simulation)
    - ML baselines (LSTM, CNN, etc.)
    """

    def __init__(self):
        """Initialize baseline comparator."""
        self.baselines = {}
        self.results = {}

    def add_baseline(self, name: str, baseline: Any):
        """
        Add a baseline method.

        Parameters:
        -----------
        name : str
            Baseline name
        baseline : object
            Baseline object (EACBaseline, TDSBaseline, MLBaseline, etc.)
        """
        self.baselines[name] = baseline

    def compare_trajectory_prediction(
        self, test_cases: List[Dict], pinn_model: Any, metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Compare trajectory prediction across baselines.

        Parameters:
        -----------
        test_cases : list
            List of test case dictionaries with parameters
        pinn_model : object
            Trained PINN model
        metrics : list, optional
            List of metric names to compute

        Returns:
        --------
        results : dict
            Dictionary of results for each baseline
        """
        if metrics is None:
            metrics = ["rmse", "mae", "mape", "r2"]

        results = {}

        # Test PINN
        pinn_results = self._evaluate_pinn(pinn_model, test_cases, metrics)
        results["PINN"] = pinn_results

        # Test each baseline
        for name, baseline in self.baselines.items():
            baseline_results = self._evaluate_baseline(baseline, test_cases, metrics)
            results[name] = baseline_results

        self.results["trajectory"] = results
        return results

    def compare_cct_estimation(
        self, test_cases: List[Dict], pinn_model: Any, true_cct: Optional[np.ndarray] = None
    ) -> Dict[str, Dict]:
        """
        Compare CCT estimation across baselines.

        Parameters:
        -----------
        test_cases : list
            List of test case dictionaries
        pinn_model : object
            Trained PINN model
        true_cct : np.ndarray, optional
            True CCT values for comparison

        Returns:
        --------
        results : dict
            Dictionary of results
        """
        results = {}

        # Extract parameters from test cases
        cct_predictions = {}

        # PINN CCT (using binary search)
        # This would use utils.cct_binary_search
        # For now, placeholder
        cct_predictions["PINN"] = None

        # EAC baseline
        if "EAC" in self.baselines:
            eac = self.baselines["EAC"]
            cct_eac = []
            for case in test_cases:
                cct = eac.estimate_cct(
                    Pm=case["Pm"],
                    M=case["M"],
                    D=case["D"],
                    Xprefault=case["Xprefault"],
                    Xfault=case["Xfault"],
                    Xpostfault=case["Xpostfault"],
                    V1=case.get("V1", 1.05),
                    V2=case.get("V2", 1.0),
                )
                cct_eac.append(cct)
            cct_predictions["EAC"] = np.array(cct_eac)

        # TDS baseline
        if "TDS" in self.baselines:
            tds = self.baselines["TDS"]
            cct_tds = []
            for case in test_cases:
                cct = tds.find_cct(Pm=case["Pm"], M=case["M"], D=case["D"])
                cct_tds.append(cct)
            cct_predictions["TDS"] = np.array(cct_tds)

        # Compute metrics
        if true_cct is not None:
            for name, pred in cct_predictions.items():
                if pred is not None:
                    metrics = compute_cct_metrics(pred, true_cct)
                    results[name] = metrics

        self.results["cct"] = results
        return results

    def _evaluate_pinn(self, pinn_model: Any, test_cases: List[Dict], metrics: List[str]) -> Dict:
        """Evaluate PINN model."""
        # Placeholder - would integrate with actual PINN prediction
        return {}

    def _evaluate_baseline(self, baseline: Any, test_cases: List[Dict], metrics: List[str]) -> Dict:
        """Evaluate a baseline method."""
        # Placeholder - would integrate with baseline prediction
        return {}

    def generate_comparison_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive comparison report.

        Parameters:
        -----------
        output_file : str, optional
            File path to save report

        Returns:
        --------
        report : str
            Formatted report string
        """
        report = "=" * 80 + "\n"
        report += "BASELINE COMPARISON REPORT\n"
        report += "=" * 80 + "\n\n"

        # Trajectory prediction results
        if "trajectory" in self.results:
            report += "TRAJECTORY PREDICTION RESULTS\n"
            report += "-" * 80 + "\n"
            for method, metrics in self.results["trajectory"].items():
                report += "\n{method}:\n"
                for metric, value in metrics.items():
                    report += "  {metric}: {value:.6f}\n"
            report += "\n"

        # CCT estimation results
        if "cct" in self.results:
            report += "CCT ESTIMATION RESULTS\n"
            report += "-" * 80 + "\n"
            for method, metrics in self.results["cct"].items():
                report += "\n{method}:\n"
                for metric, value in metrics.items():
                    report += "  {metric}: {value:.6f}\n"
            report += "\n"

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)

        return report
