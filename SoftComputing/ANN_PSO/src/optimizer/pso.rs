use rand::Rng;

/// PSO configuration parameters
pub struct PsoConfig {
    pub num_particles: usize,
    pub dim: usize,
    pub w: f64,       // Inertia weight
    pub c1: f64,      // Cognitive coefficient
    pub c2: f64,      // Social coefficient
    pub pos_min: f64, // Position bounds
    pub pos_max: f64,
    pub vel_max: f64, // Velocity clamp
}

impl Default for PsoConfig {
    fn default() -> Self {
        Self {
            num_particles: 30,
            dim: 9,
            w: 0.7,
            c1: 1.5,
            c2: 1.5,
            pos_min: -10.0,
            pos_max: 10.0,
            vel_max: 1.0,
        }
    }
}

/// A single particle in the swarm
struct Particle {
    position: Vec<f64>,
    velocity: Vec<f64>,
    best_position: Vec<f64>,
    best_fitness: f64,
}

/// Particle Swarm Optimization
pub struct Pso {
    particles: Vec<Particle>,
    global_best_position: Vec<f64>,
    global_best_fitness: f64,
    config: PsoConfig,
}

impl Pso {
    /// Create a new PSO optimizer
    pub fn new(config: PsoConfig) -> Self {
        let mut rng = rand::thread_rng();
        let mut particles = Vec::with_capacity(config.num_particles);
        let global_best_position = vec![0.0; config.dim];
        let global_best_fitness = f64::INFINITY;

        for _ in 0..config.num_particles {
            // Initialize random position
            let position: Vec<f64> = (0..config.dim)
                .map(|_| rng.gen_range(config.pos_min..=config.pos_max))
                .collect();

            // Initialize velocity to small random values
            let velocity: Vec<f64> = (0..config.dim)
                .map(|_| rng.gen_range(-config.vel_max..=config.vel_max) * 0.1)
                .collect();

            let particle = Particle {
                position: position.clone(),
                velocity,
                best_position: position,
                best_fitness: f64::INFINITY,
            };
            particles.push(particle);
        }

        Self {
            particles,
            global_best_position,
            global_best_fitness,
            config,
        }
    }

    /// Perform one PSO iteration
    /// fitness_fn: lower is better (minimization)
    pub fn step<F>(&mut self, mut fitness_fn: F)
    where
        F: FnMut(&[f64]) -> f64,
    {
        let mut rng = rand::thread_rng();

        for particle in &mut self.particles {
            // Update velocity
            for d in 0..self.config.dim {
                let r1: f64 = rng.gen();
                let r2: f64 = rng.gen();

                particle.velocity[d] = self.config.w * particle.velocity[d]
                    + self.config.c1 * r1 * (particle.best_position[d] - particle.position[d])
                    + self.config.c2 * r2 * (self.global_best_position[d] - particle.position[d]);

                // Clamp velocity
                particle.velocity[d] = particle.velocity[d]
                    .clamp(-self.config.vel_max, self.config.vel_max);
            }

            // Update position
            for d in 0..self.config.dim {
                particle.position[d] += particle.velocity[d];
                // Clamp position
                particle.position[d] = particle.position[d]
                    .clamp(self.config.pos_min, self.config.pos_max);
            }

            // Evaluate fitness
            let fitness = fitness_fn(&particle.position);

            // Update personal best
            if fitness < particle.best_fitness {
                particle.best_fitness = fitness;
                particle.best_position = particle.position.clone();
            }

            // Update global best
            if fitness < self.global_best_fitness {
                self.global_best_fitness = fitness;
                self.global_best_position = particle.position.clone();
            }
        }
    }

    /// Get the best position found
    pub fn best_position(&self) -> &[f64] {
        &self.global_best_position
    }

    /// Get the best fitness found
    pub fn best_fitness(&self) -> f64 {
        self.global_best_fitness
    }

    /// Initialize particles and evaluate initial fitness
    pub fn init<F>(&mut self, mut fitness_fn: F)
    where
        F: FnMut(&[f64]) -> f64,
    {
        for particle in &mut self.particles {
            let fitness = fitness_fn(&particle.position);
            particle.best_fitness = fitness;

            if fitness < self.global_best_fitness {
                self.global_best_fitness = fitness;
                self.global_best_position = particle.position.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pso_sphere_function() {
        // Minimize f(x) = sum(x^2), optimal at x = 0
        let config = PsoConfig {
            num_particles: 20,
            dim: 2,
            w: 0.7,
            c1: 1.5,
            c2: 1.5,
            pos_min: -10.0,
            pos_max: 10.0,
            vel_max: 1.0,
        };

        let mut pso = Pso::new(config);
        pso.init(|x| x.iter().map(|v| v * v).sum());

        for _ in 0..100 {
            pso.step(|x| x.iter().map(|v| v * v).sum());
        }

        // Should converge close to 0
        assert!(pso.best_fitness() < 0.1);
    }

    #[test]
    fn test_pso_initialization() {
        let config = PsoConfig {
            num_particles: 10,
            dim: 5,
            w: 0.7,
            c1: 1.5,
            c2: 1.5,
            pos_min: -5.0,
            pos_max: 5.0,
            vel_max: 1.0,
        };

        let pso = Pso::new(config);

        // Verify particle count
        assert_eq!(pso.particles.len(), 10);

        // Verify all particles have correct dimension
        for particle in &pso.particles {
            assert_eq!(particle.position.len(), 5);
            assert_eq!(particle.velocity.len(), 5);
            assert_eq!(particle.best_position.len(), 5);

            // Verify positions are within bounds
            for &pos in &particle.position {
                assert!(pos >= -5.0 && pos <= 5.0);
            }

            // Verify velocities are within initial range (0.1 * vel_max)
            for &vel in &particle.velocity {
                assert!(vel >= -0.1 && vel <= 0.1);
            }
        }
    }

    #[test]
    fn test_pso_velocity_clamp() {
        let config = PsoConfig {
            num_particles: 5,
            dim: 2,
            w: 0.7,
            c1: 2.0,
            c2: 2.0,
            pos_min: -10.0,
            pos_max: 10.0,
            vel_max: 0.5, // Small vel_max to trigger clamping
        };

        let mut pso = Pso::new(config);

        // Use a function that creates large gradients
        pso.init(|x| x[0] * 100.0);

        for _ in 0..10 {
            pso.step(|x| x[0] * 100.0);
        }

        // Verify all velocities are clamped
        for particle in &pso.particles {
            for &vel in &particle.velocity {
                assert!(
                    vel >= -0.5 && vel <= 0.5,
                    "Velocity {} exceeds bounds",
                    vel
                );
            }
        }
    }

    #[test]
    fn test_pso_position_bounds() {
        let config = PsoConfig {
            num_particles: 5,
            dim: 2,
            w: 0.9,
            c1: 2.0,
            c2: 2.0,
            pos_min: -2.0,
            pos_max: 2.0,
            vel_max: 5.0, // Large velocity to push particles out of bounds
        };

        let mut pso = Pso::new(config);

        // Use a function that pulls particles to extreme values
        pso.init(|x| -x[0] - x[1]); // Minimize by going to +infinity

        for _ in 0..50 {
            pso.step(|x| -x[0] - x[1]);
        }

        // Verify all positions are clamped to bounds
        for particle in &pso.particles {
            for &pos in &particle.position {
                assert!(
                    pos >= -2.0 && pos <= 2.0,
                    "Position {} exceeds bounds",
                    pos
                );
            }
        }
    }

    #[test]
    fn test_pso_pbest_update() {
        let config = PsoConfig {
            num_particles: 1,
            dim: 2,
            w: 0.0,  // No inertia
            c1: 0.0, // No cognitive component
            c2: 0.0, // No social component
            pos_min: -10.0,
            pos_max: 10.0,
            vel_max: 1.0,
        };

        let mut pso = Pso::new(config);

        // Track fitness values
        let mut call_count = 0;

        // First call: high fitness (bad)
        pso.init(|_| {
            call_count += 1;
            100.0
        });

        let initial_pbest = pso.particles[0].best_fitness;
        assert_eq!(initial_pbest, 100.0);

        // Next iteration: better fitness
        pso.step(|_| 50.0);
        assert_eq!(pso.particles[0].best_fitness, 50.0);

        // Next iteration: worse fitness (should not update pbest)
        pso.step(|_| 80.0);
        assert_eq!(pso.particles[0].best_fitness, 50.0);

        // Next iteration: even better fitness
        pso.step(|_| 10.0);
        assert_eq!(pso.particles[0].best_fitness, 10.0);
    }

    #[test]
    fn test_pso_gbest_update() {
        let config = PsoConfig {
            num_particles: 3,
            dim: 1,
            w: 0.0,
            c1: 0.0,
            c2: 0.0,
            pos_min: -10.0,
            pos_max: 10.0,
            vel_max: 0.0, // No movement
        };

        let mut pso = Pso::new(config);

        // Initialize with different fitness values
        let mut particle_idx = 0;
        pso.init(|_| {
            let fitness = match particle_idx {
                0 => 30.0,
                1 => 10.0, // Best
                2 => 20.0,
                _ => 100.0,
            };
            particle_idx += 1;
            fitness
        });

        // Global best should be 10.0 (from particle 1)
        assert_eq!(pso.best_fitness(), 10.0);
    }

    #[test]
    fn test_pso_rastrigin() {
        // Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
        // Global minimum at x = 0 with f(0) = 0
        fn rastrigin(x: &[f64]) -> f64 {
            let n = x.len() as f64;
            let sum: f64 = x
                .iter()
                .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum();
            10.0 * n + sum
        }

        let config = PsoConfig {
            num_particles: 50,
            dim: 2,
            w: 0.729,
            c1: 1.49445,
            c2: 1.49445,
            pos_min: -5.12,
            pos_max: 5.12,
            vel_max: 1.0,
        };

        let mut pso = Pso::new(config);
        pso.init(rastrigin);

        for _ in 0..200 {
            pso.step(rastrigin);
        }

        // Rastrigin is harder, so we use a looser threshold
        // Should find a reasonably good solution (not necessarily global optimum)
        assert!(
            pso.best_fitness() < 5.0,
            "PSO failed to optimize Rastrigin, got {}",
            pso.best_fitness()
        );
    }
}
