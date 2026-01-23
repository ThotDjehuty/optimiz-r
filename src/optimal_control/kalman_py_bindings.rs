//! Python bindings for Kalman Filter module

use super::{
    FilterResult, KalmanFilter, KalmanState, LinearObservation, LinearStateTransition,
    RTSSmoother, SmootherResult, UnscentedKalmanFilter,
};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;

/// Python wrapper for KalmanState
#[pyclass(name = "KalmanState")]
#[derive(Clone)]
pub struct PyKalmanState {
    pub inner: KalmanState,
}

#[pymethods]
impl PyKalmanState {
    #[new]
    fn new(state: Vec<f64>, covariance: Vec<Vec<f64>>) -> PyResult<Self> {
        let state_arr = Array1::from_vec(state);
        let n = state_arr.len();

        let flat_cov: Vec<f64> = covariance.into_iter().flatten().collect();
        let cov_arr = Array2::from_shape_vec((n, n), flat_cov)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(Self {
            inner: KalmanState::new(state_arr, cov_arr),
        })
    }

    /// Get state estimate
    fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.state.to_pyarray_bound(py)
    }

    /// Get covariance matrix
    fn get_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.covariance.to_pyarray_bound(py)
    }

    /// Get innovation (residual)
    fn get_innovation<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .innovation
            .as_ref()
            .map(|inn| inn.to_pyarray_bound(py))
    }

    /// Get innovation covariance
    fn get_innovation_covariance<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .innovation_covariance
            .as_ref()
            .map(|cov| cov.to_pyarray_bound(py))
    }

    /// Get log-likelihood
    fn get_log_likelihood(&self) -> Option<f64> {
        self.inner.log_likelihood
    }

    fn __repr__(&self) -> String {
        format!(
            "KalmanState(dim={}, log_likelihood={:?})",
            self.inner.dim(),
            self.inner.log_likelihood
        )
    }
}

/// Python wrapper for LinearKalmanFilter
#[pyclass(name = "LinearKalmanFilter")]
pub struct PyLinearKalmanFilter {
    filter: KalmanFilter<LinearStateTransition, LinearObservation>,
}

#[pymethods]
impl PyLinearKalmanFilter {
    #[new]
    #[pyo3(signature = (f_matrix, h_matrix, q_matrix, r_matrix, initial_state, initial_covariance, b_matrix=None))]
    fn new(
        f_matrix: Vec<Vec<f64>>,
        h_matrix: Vec<Vec<f64>>,
        q_matrix: Vec<Vec<f64>>,
        r_matrix: Vec<Vec<f64>>,
        initial_state: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
        b_matrix: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        // Convert F matrix
        let n_state = f_matrix.len();
        let f_flat: Vec<f64> = f_matrix.into_iter().flatten().collect();
        let f = Array2::from_shape_vec((n_state, n_state), f_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // Convert H matrix
        let n_obs = h_matrix.len();
        let h_flat: Vec<f64> = h_matrix.into_iter().flatten().collect();
        let h = Array2::from_shape_vec((n_obs, n_state), h_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // Convert Q matrix
        let q_flat: Vec<f64> = q_matrix.into_iter().flatten().collect();
        let q = Array2::from_shape_vec((n_state, n_state), q_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // Convert R matrix
        let r_flat: Vec<f64> = r_matrix.into_iter().flatten().collect();
        let r = Array2::from_shape_vec((n_obs, n_obs), r_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // Convert B matrix if provided
        let b = if let Some(b_mat) = b_matrix {
            let n_control = b_mat[0].len();
            let b_flat: Vec<f64> = b_mat.into_iter().flatten().collect();
            Some(
                Array2::from_shape_vec((n_state, n_control), b_flat).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                })?,
            )
        } else {
            None
        };

        // Create state and observation models
        let state_model = LinearStateTransition {
            f_matrix: f,
            b_matrix: b,
            q_matrix: q,
        };

        let obs_model = LinearObservation {
            h_matrix: h,
            r_matrix: r,
        };

        // Create initial state
        let state_arr = Array1::from_vec(initial_state);
        let cov_flat: Vec<f64> = initial_covariance.into_iter().flatten().collect();
        let cov_arr = Array2::from_shape_vec((n_state, n_state), cov_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let initial = KalmanState::new(state_arr, cov_arr);

        Ok(Self {
            filter: KalmanFilter::new(state_model, obs_model, initial),
        })
    }

    /// Get current state
    fn get_state(&self) -> PyKalmanState {
        PyKalmanState {
            inner: self.filter.state().clone(),
        }
    }

    /// Prediction step
    #[pyo3(signature = (control=None))]
    fn predict(&mut self, control: Option<Vec<f64>>) -> PyResult<()> {
        let control_arr = control.as_ref().map(|c| Array1::from_vec(c.clone()));
        self.filter
            .predict(control_arr.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Update step
    fn update(&mut self, observation: Vec<f64>) -> PyResult<()> {
        let obs_arr = Array1::from_vec(observation);
        self.filter
            .update(&obs_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Run filter on sequence of observations
    #[pyo3(signature = (observations, controls=None))]
    fn filter(
        &mut self,
        observations: Vec<Vec<f64>>,
        controls: Option<Vec<Vec<f64>>>,
    ) -> PyResult<PyFilterResult> {
        let obs_arrays: Vec<Array1<f64>> = observations
            .into_iter()
            .map(Array1::from_vec)
            .collect();

        let control_arrays: Option<Vec<Array1<f64>>> = controls
            .map(|c| c.into_iter().map(Array1::from_vec).collect());

        let result = self
            .filter
            .filter(&obs_arrays, control_arrays.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(PyFilterResult { inner: result })
    }

    fn __repr__(&self) -> String {
        format!(
            "LinearKalmanFilter(state_dim={})",
            self.filter.state().dim()
        )
    }
}

/// Python wrapper for FilterResult
#[pyclass(name = "FilterResult")]
pub struct PyFilterResult {
    inner: FilterResult,
}

#[pymethods]
impl PyFilterResult {
    /// Get filtered states
    fn get_states<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<f64>>> {
        self.inner
            .states
            .iter()
            .map(|s| s.to_pyarray_bound(py))
            .collect()
    }

    /// Get filtered covariances
    fn get_covariances<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.inner
            .covariances
            .iter()
            .map(|c| c.to_pyarray_bound(py))
            .collect()
    }

    /// Get log-likelihood
    fn get_log_likelihood(&self) -> f64 {
        self.inner.log_likelihood
    }

    fn __repr__(&self) -> String {
        format!(
            "FilterResult(n_states={}, log_likelihood={:.4})",
            self.inner.states.len(),
            self.inner.log_likelihood
        )
    }
}

/// Python wrapper for RTSSmoother
#[pyclass(name = "RTSSmoother")]
pub struct PyRTSSmoother {
    state_model: LinearStateTransition,
}

#[pymethods]
impl PyRTSSmoother {
    #[new]
    #[pyo3(signature = (f_matrix, q_matrix, b_matrix=None))]
    fn new(
        f_matrix: Vec<Vec<f64>>,
        q_matrix: Vec<Vec<f64>>,
        b_matrix: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        let n_state = f_matrix.len();
        let f_flat: Vec<f64> = f_matrix.into_iter().flatten().collect();
        let f = Array2::from_shape_vec((n_state, n_state), f_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let q_flat: Vec<f64> = q_matrix.into_iter().flatten().collect();
        let q = Array2::from_shape_vec((n_state, n_state), q_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let b = if let Some(b_mat) = b_matrix {
            let n_control = b_mat[0].len();
            let b_flat: Vec<f64> = b_mat.into_iter().flatten().collect();
            Some(
                Array2::from_shape_vec((n_state, n_control), b_flat).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                })?,
            )
        } else {
            None
        };

        Ok(Self {
            state_model: LinearStateTransition {
                f_matrix: f,
                b_matrix: b,
                q_matrix: q,
            },
        })
    }

    /// Smooth filtered estimates
    fn smooth(&self, filter_result: &PyFilterResult) -> PyResult<PySmootherResult> {
        let smoother = RTSSmoother::new(self.state_model.clone());
        let result = smoother
            .smooth(&filter_result.inner)
            .map_err(|e: crate::optimal_control::OptimalControlError| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?;

        Ok(PySmootherResult { inner: result })
    }
}

/// Python wrapper for SmootherResult
#[pyclass(name = "SmootherResult")]
pub struct PySmootherResult {
    inner: SmootherResult,
}

#[pymethods]
impl PySmootherResult {
    /// Get smoothed states
    fn get_states<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<f64>>> {
        self.inner
            .states
            .iter()
            .map(|s| s.to_pyarray_bound(py))
            .collect()
    }

    /// Get smoothed covariances
    fn get_covariances<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.inner
            .covariances
            .iter()
            .map(|c| c.to_pyarray_bound(py))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("SmootherResult(n_states={})", self.inner.states.len())
    }
}

/// Python wrapper for UnscentedKalmanFilter
#[pyclass(name = "UnscentedKalmanFilter")]
pub struct PyUnscentedKalmanFilter {
    filter: UnscentedKalmanFilter<LinearStateTransition, LinearObservation>,
}

#[pymethods]
impl PyUnscentedKalmanFilter {
    #[new]
    #[pyo3(signature = (f_matrix, h_matrix, q_matrix, r_matrix, initial_state, initial_covariance, alpha=1e-3, beta=2.0, kappa=0.0, b_matrix=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        f_matrix: Vec<Vec<f64>>,
        h_matrix: Vec<Vec<f64>>,
        q_matrix: Vec<Vec<f64>>,
        r_matrix: Vec<Vec<f64>>,
        initial_state: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
        alpha: f64,
        beta: f64,
        kappa: f64,
        b_matrix: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        // Convert matrices (similar to LinearKalmanFilter)
        let n_state = f_matrix.len();
        let f_flat: Vec<f64> = f_matrix.into_iter().flatten().collect();
        let f = Array2::from_shape_vec((n_state, n_state), f_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let n_obs = h_matrix.len();
        let h_flat: Vec<f64> = h_matrix.into_iter().flatten().collect();
        let h = Array2::from_shape_vec((n_obs, n_state), h_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let q_flat: Vec<f64> = q_matrix.into_iter().flatten().collect();
        let q = Array2::from_shape_vec((n_state, n_state), q_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let r_flat: Vec<f64> = r_matrix.into_iter().flatten().collect();
        let r = Array2::from_shape_vec((n_obs, n_obs), r_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let b = if let Some(b_mat) = b_matrix {
            let n_control = b_mat[0].len();
            let b_flat: Vec<f64> = b_mat.into_iter().flatten().collect();
            Some(
                Array2::from_shape_vec((n_state, n_control), b_flat).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                })?,
            )
        } else {
            None
        };

        let state_model = LinearStateTransition {
            f_matrix: f,
            b_matrix: b,
            q_matrix: q,
        };

        let obs_model = LinearObservation {
            h_matrix: h,
            r_matrix: r,
        };

        let state_arr = Array1::from_vec(initial_state);
        let cov_flat: Vec<f64> = initial_covariance.into_iter().flatten().collect();
        let cov_arr = Array2::from_shape_vec((n_state, n_state), cov_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let initial = KalmanState::new(state_arr, cov_arr);

        Ok(Self {
            filter: UnscentedKalmanFilter::new(state_model, obs_model, initial, alpha, beta, kappa),
        })
    }

    /// Get current state
    fn get_state(&self) -> PyKalmanState {
        PyKalmanState {
            inner: self.filter.state().clone(),
        }
    }

    /// Prediction step
    #[pyo3(signature = (control=None))]
    fn predict(&mut self, control: Option<Vec<f64>>) -> PyResult<()> {
        let control_arr = control.as_ref().map(|c| Array1::from_vec(c.clone()));
        self.filter
            .predict(control_arr.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Update step
    fn update(&mut self, observation: Vec<f64>) -> PyResult<()> {
        let obs_arr = Array1::from_vec(observation);
        self.filter
            .update(&obs_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "UnscentedKalmanFilter(state_dim={})",
            self.filter.state().dim()
        )
    }
}

/// Register Python functions for Kalman filter
pub fn register_python_functions(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyKalmanState>()?;
    m.add_class::<PyLinearKalmanFilter>()?;
    m.add_class::<PyFilterResult>()?;
    m.add_class::<PyRTSSmoother>()?;
    m.add_class::<PySmootherResult>()?;
    m.add_class::<PyUnscentedKalmanFilter>()?;
    Ok(())
}
