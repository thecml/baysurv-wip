import tensorflow as tf
import tensorflow_probability as tfp

@tf.function
def sample_hmc(log_prob, inits, n_steps, n_burnin_steps, bijectors_list = None):
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob,
        step_size=0.1,
        num_leapfrog_steps=2
    )

    if bijectors_list is not None:
        inner_kernel = tfp.mcmc.TransformedTransitionKernel(inner_kernel, bijectors_list)

    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel,
        num_adaptation_steps=int(n_burnin_steps * 0.8)
    )

    return tfp.mcmc.sample_chain(
        num_results=n_steps,
        current_state=inits,
        kernel=adaptive_kernel,
        num_burnin_steps=n_burnin_steps,
        trace_fn=lambda _, pkr: [pkr.inner_results.is_accepted,
                                 pkr.inner_results.log_accept_ratio]
    )