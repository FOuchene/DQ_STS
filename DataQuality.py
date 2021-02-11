import matplotlib.dates as mdates
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability import sts
from matplotlib import pylab as plt
import Charting
import datetime

tf.enable_v2_behavior()


def build_model(Risk_Factors):
    # Seasonality effect if relevant: daily data assumed here, datetime index assumed to be comprised of business days
    day_of_week_effect = sts.Seasonal(
        num_seasons=5,
        observed_time_series=np.array(Risk_Factors[Risk_Factors.columns[1]]).astype(np.float32),
        name='day_of_week_effect')

    # effect of each explicative factor
    data = np.array(Risk_Factors[Risk_Factors.columns[2:]]).astype(np.float32)
    other_effects = sts.LinearRegression(design_matrix=tf.reshape(data-np.mean(data), (-1, data.shape[1])),
                                            name='other_effects')
    # auto-regressive effect
    autoregressive = sts.Autoregressive(
        order=1,
        observed_time_series=np.array(Risk_Factors[Risk_Factors.columns[1]]).astype(np.float32),
        name='autoregressive')

    model = sts.Sum([day_of_week_effect, other_effects, autoregressive],
                    observed_time_series=np.array(Risk_Factors[Risk_Factors.columns[1]]).astype(np.float32))
    return model


class Forecast:
    def __init__(self, RiskFactors, num_forecast_steps):
        # Input DataFrame Format: columns: Datetime,Time Serie, Explicative Factors 1...n
        self.risk_factors = RiskFactors
        self.num_forecast_steps = num_forecast_steps
        self.data_loc = mdates.WeekdayLocator(byweekday=mdates.WE, interval=4)
        self.data_fmt = mdates.DateFormatter('%Y-%b-%d')
        self.Model = build_model(self.risk_factors)
        self.dataset = np.array(self.risk_factors[self.risk_factors.columns[1]]).astype(np.float32)
        self.dates =np.array(self.risk_factors[self.risk_factors.columns[0]]).astype('datetime64[D]')
        self.training_data = self.dataset[:-self.num_forecast_steps]

    def __call__(self):
        variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=self.Model)

        num_variational_steps = 200  # @param { isTemplate: true}
        optimizer = tf.optimizers.Adam(learning_rate=.1)

        # Using fit_surrogate_posterior to build and optimize the variational loss function.
        elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=self.Model.joint_log_prob(observed_time_series=self.training_data),
            surrogate_posterior=variational_posteriors,
            optimizer=optimizer,
            num_steps=num_variational_steps)

        # plot elbo loss curve to check model convergence
        plt.plot(elbo_loss_curve)
        plt.title('elbo loss curve')
        plt.show()

        # Draw samples from the variational posterior
        q_samples_ = variational_posteriors.sample(50)

        # Forecasting
        forecast_dist = tfp.sts.forecast(
            model=self.Model,
            observed_time_series=self.training_data,
            parameter_samples=q_samples_,
            num_steps_forecast=self.num_forecast_steps)
        num_samples = self.num_forecast_steps

        (
            forecast_mean,
            forecast_scale,
            forecast_samples
        ) = (
            forecast_dist.mean().numpy()[..., 0],
            forecast_dist.stddev().numpy()[..., 0],
            forecast_dist.sample(num_samples).numpy()[..., 0]
        )
        # plot forecast vs actual data
        fig, ax = Charting.plot_forecast(self.dates, self.dataset,
                                         forecast_mean, forecast_scale, forecast_samples,
                                         title="Time serie forecast", x_locator=self.data_loc,
                                         x_formatter=self.data_fmt)
        fig.tight_layout()

        one_step_dist = sts.one_step_predictive(
            self.Model,
            observed_time_series=self.dataset,
            parameter_samples=q_samples_)

        one_step_mean, one_step_scale = (one_step_dist.mean().numpy(),
                                         one_step_dist.stddev().numpy())

        fig, ax = Charting.plot_one_step_predictive(self.dates, self.dataset,
                                                    one_step_mean, one_step_scale,
                                                    x_locator=self.data_loc, x_formatter=self.data_fmt)

        # one-step-ahead forecasts to detect anomalous data
        zscores = np.abs((self.dataset - one_step_mean) / one_step_scale)
        anomalies = zscores > 1.96
        ax.scatter(self.dates[anomalies],
                   self.dataset[anomalies],
                   c="red", marker="x", s=20, linewidth=2, label=r"Outlier")
        ax.legend()
        plt.show()
