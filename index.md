### Theoretical Physics MSci @ University of Nottingham Undergraduate

## Featured Research and Projects

### **N-body Graviational Dynamics Solver** - 3rd Year University Scientific Computing Project

<details>

<summary> N-body Project </summary>

```python
    @nb.njit
    def differential_system_N_Optimised(t, state, m, G):
        """
        Sets up the coupled solution in terms of matrices to be run in numpy to be as fast as possible

        """
        positions = state.reshape(-1, 6)[:, POSITION_COLS]
        velocities = state.reshape(-1, 6)[:, VELOCITY_COLS]
        N = len(m)
        
        position_differences = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        
        r2 = np.sum(position_differences**2, axis = 2)
        
        np.fill_diagonal(r2, 1)
        m_j = m[np.newaxis, :]
        
        
        coeff = (m_j) / ((r2)**1.5)
            
        np.fill_diagonal(coeff, 0.0)
        
        A = G * np.sum(position_differences * coeff[:, :, np.newaxis], axis=1)

        
        d_state = np.empty(6 * N, dtype=np.float64)
        d_state[0::6] = velocities[:, 0]
        d_state[3::6] = A[:, 0]
        d_state[1::6] = velocities[:, 1]
        d_state[4::6] = A[:, 1]
        d_state[2::6] = velocities[:, 2]
        d_state[5::6] = A[:, 2]
        
        return d_state
```

<\details>