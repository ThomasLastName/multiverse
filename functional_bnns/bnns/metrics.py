import torch
from bnns.utils.math import iqr, cor, univar_poly_fit
from bnns.utils.handling import my_warn
from bnns.utils.plotting import points_with_curves


### ~~~
## ~~~ All metrics with a `predictions` argument assume predictions.shape==( N_POSTERIOR_SAMPLES, n_test, n_out_features )
### ~~~


#
# ~~~ Measure MSE of a deterministic model with error=model(x_test)-y_test
def rmse(model, x_test, y_test):
    with torch.no_grad():
        pred = model(x_test)
        assert pred.shape == y_test.shape
        return ((pred - y_test) ** 2).mean().sqrt().item()


#
# ~~~ Measure MAE of a deterministic model with error=model(x_test)-y_test
def mae(model, x_test, y_test):
    with torch.no_grad():
        pred = model(x_test)
        assert pred.shape == y_test.shape
        return (pred - y_test).abs().mean().item()


#
# ~~~ Measure max norm of a deterministic model with error=model(x_test)-y_test
def max_norm(model, x_test, y_test):
    with torch.no_grad():
        pred = model(x_test)
        assert pred.shape == y_test.shape
        return (pred - y_test).abs().max().item()


#
# ~~~ Measure MSE of the predictive median
def rmse_of_median(predictions, y_test):
    with torch.no_grad():
        pred = predictions.median(dim=0).values
        assert pred.shape == y_test.shape
        return ((pred - y_test) ** 2).mean().sqrt().item()


#
# ~~~ Measure MAE of the predictive median
def mae_of_median(predictions, y_test):
    with torch.no_grad():
        pred = predictions.median(dim=0).values
        assert pred.shape == y_test.shape
        return (pred - y_test).abs().mean().item()


#
# ~~~ Measure max norm of the predictive median
def max_norm_of_median(predictions, y_test):
    with torch.no_grad():
        pred = predictions.median(dim=0).values
        assert pred.shape == y_test.shape
        return (pred - y_test).abs().max().item()


#
# ~~~ Measure MSE of the predictive mean
def rmse_of_mean(predictions, y_test):
    with torch.no_grad():
        pred = predictions.mean(dim=0)
        assert pred.shape == y_test.shape
        return ((pred - y_test) ** 2).mean().sqrt().item()


#
# ~~~ Measure MAE of the predictive mean
def mae_of_mean(predictions, y_test):
    with torch.no_grad():
        pred = predictions.mean(dim=0)
        assert pred.shape == y_test.shape
        return (pred - y_test).abs().mean().item()


#
# ~~~ Measure max norm of the predictive mean
def max_norm_of_mean(predictions, y_test):
    with torch.no_grad():
        pred = predictions.mean(dim=0)
        assert pred.shape == y_test.shape
        return (pred - y_test).abs().max().item()


#
# ~~~ Compute what would be the middle (50% percentile) line of the box plots in figure 4 of the SLOSH paper
def median_norm_error(predictions, y_test):
    n_posterior_samples = predictions.shape[0]
    residuals = predictions - torch.tile(y_test, (n_posterior_samples, 1, 1))
    errors = residuals.norm(dim=-1)  # ~~~ has shape (N_POSTERIOR_SAMPLES,n_test)
    rMSE_of_the_posterior_samples = errors.norm(
        dim=-0
    )  # ~~~ figure 4 in the SLOSH paper shows a box plot of these
    return (
        rMSE_of_the_posterior_samples.meadian()
    )  # ~~~ this is the middle line of said box plot


# #
# # ~~~ Measure MSE of the predictive median relative
# def median_of_mse( predictions, y_test ):
#     with torch.no_grad():
#         y_test = y_test.flatten()
#         pred = predictions.T
#         assert pred.shape[1] == y_test.shape[0]
#         return (( pred - y_test )**2).mean(dim=0).median().values

# #
# # ~~~ Measure MSE of the predictive median relative
# def mean_of_mse( predictions, y_test ):
#     with torch.no_grad():
#         y_test = y_test.flatten()
#         pred = predictions.T
#         assert pred.shape[1] == y_test.shape[0]
#         return (( pred - y_test )**2).mean()


#
# ~~~ Measure strength of the relation "predictive uncertainty (std. dev. / iqr)" ~ "accuracy (MSE of the predictive mean / predictive median)"
def uncertainty_vs_accuracy(
    predictions,
    y_test,
    quantile_uncertainty,
    quantile_accuracy,
    show=True,
    verbose=True,
):
    with torch.no_grad():
        uncertainty = (
            iqr(predictions, dim=0) if quantile_uncertainty else predictions.std(dim=0)
        )
        point_estimate = (
            predictions.median(dim=0).values
            if quantile_accuracy
            else predictions.mean(dim=0)
        )
        accuracy = (point_estimate - y_test) ** 2
        assert (
            y_test.shape == point_estimate.shape == uncertainty.shape == accuracy.shape
        )
        uncertainty = uncertainty.flatten().cpu().numpy()
        accuracy = accuracy.flatten().cpu().numpy()
        #
        # ~~~ Use polynomial regression to measure the strength of the relation uncertainty~accuracy
        fits = [univar_poly_fit(y=uncertainty, x=accuracy, degree=k) for k in (1, 2, 3)]
        polys = [fit[0] for fit in fits]
        R_squared_coefficients = [fit[2] for fit in fits]
        if show:
            points_with_curves(
                y=uncertainty,
                x=accuracy,
                points_label="Values on the Val./Test Set",
                marker_color="grey",
                curves=polys,
                curve_labels=[f"R^2 {val:.3}" for val in R_squared_coefficients],
                title="Uncertainty vs Accuracy of Predictions (with Polynomial Fits)",
                xlabel="Accuracy of Predictions (viz. |True Value - Posertior Predictive "
                + ("median" if quantile_accuracy else "mean")
                + "|^2)",
                ylabel=(
                    "Uncertainty (viz. Posterior Predictive " + "IQR)"
                    if quantile_uncertainty
                    else "std. dev.)"
                ),
                model_fit=False,
            )
        slope_in_OLS = fits[0][1][0]
        R_squared_of_OLS = R_squared_coefficients[0]
        if (max(R_squared_coefficients) > R_squared_of_OLS + 0.1) and verbose:
            my_warn(
                f"The OLS fit (R^2 {R_squared_of_OLS:.4}) was outperformed by a higher degree polynomial fit (R^2 {max(R_squared_coefficients):.4}). Using either beta_1 from OLS or the correlation may result in an inaccurate quantification of the relation. Please check the plot returned by the `show=True` argument."
            )
        return slope_in_OLS, cor(uncertainty, accuracy)


#
# ~~~ Measure strength of the relation "predictive uncertainty (std. dev.)" ~ "distance from training points"
def uncertainty_vs_proximity(
    predictions,
    quantile_uncertainty,
    x_test,
    x_train,
    show=True,
    title="Uncertainty vs Proximity to Data (with Polynomial Fits)",
    verbose=True,
):
    with torch.no_grad():
        uncertainty = (
            iqr(predictions, dim=0) if quantile_uncertainty else predictions.std(dim=0)
        )
        proximity = (
            torch.cdist(
                x_test.reshape(x_test.shape[0], -1),
                x_train.reshape(x_train.shape[0], -1),
            )
            .min(dim=1)
            .values
        )
        n_out_features = predictions.shape[-1]
        proximity = torch.column_stack(
            n_out_features * [proximity]
        )  # ~~~ proximity to training data is the same for each of the output features
        assert proximity.shape == uncertainty.shape
        uncertainty = uncertainty.flatten().cpu().numpy()
        proximity = proximity.flatten().cpu().numpy()
        #
        # ~~~ Use polynomial regression to measure the strength of the relation uncertainty~proximity
        fits = [
            univar_poly_fit(y=uncertainty, x=proximity, degree=k) for k in (1, 2, 3)
        ]
        polys = [fit[0] for fit in fits]
        R_squared_coefficients = [fit[2] for fit in fits]
        if show:
            points_with_curves(
                y=uncertainty,
                x=proximity,
                points_label="Values on the Val./Test Set",
                marker_color="grey",
                curves=polys,
                curve_labels=[f"R^2 {val:.3}" for val in R_squared_coefficients],
                title=title,
                xlabel="Distance to the Nearest Training Data Point",
                ylabel=(
                    "Uncertainty (viz. Posterior Predictive " + "IQR)"
                    if quantile_uncertainty
                    else "std. dev.)"
                ),
                model_fit=False,
            )
        slope_in_OLS = fits[0][1][0]
        R_squared_of_OLS = R_squared_coefficients[0]
        if (max(R_squared_coefficients) > R_squared_of_OLS + 0.1) and verbose:
            my_warn(
                f"The OLS fit (R^2 {R_squared_of_OLS:.4}) was outperformed by a higher degree polynomial fit (R^2 {max(R_squared_coefficients):.4}). Using either beta_1 from OLS or the correlation may result in an inaccurate quantification of the relation. Please check the plot returned by the `show=True` argument."
            )
        return slope_in_OLS, cor(uncertainty, proximity)


#
# ~~~ Measure how heterognenous the posterior predictive distribution is, i.e., how much spread we witness in the uncertainty levels for different inputs
def uncertainty_spread(predictions, quantile_uncertainty):
    with torch.no_grad():
        uncertainty = (
            iqr(predictions, dim=0) if quantile_uncertainty else predictions.std(dim=0)
        )
        all_uncertainty_levels = uncertainty.flatten()
        return (all_uncertainty_levels.max() - all_uncertainty_levels.min()).item()


#
# ~~~ Measure how heterognenous the posterior predictive distribution is, i.e., how much spread we witness in the uncertainty levels for different inputs
def uncertainty_min(predictions, quantile_uncertainty):
    with torch.no_grad():
        uncertainty = (
            iqr(predictions, dim=0) if quantile_uncertainty else predictions.std(dim=0)
        )
        all_uncertainty_levels = uncertainty.flatten()
        return all_uncertainty_levels.min().item()


#
# ~~~ Compute the interval score for all predictions "averaged over the spatial domain" (dim=1) instead of over the testing dataset (dim=0)
def avg_interval_score_of_response_features(
    predictions, y_test, quantile_uncertainty, alpha=0.95
):
    with torch.no_grad():
        if quantile_uncertainty:
            lo, hi = predictions.quantile(
                q=torch.Tensor([(1 - alpha) / 2, alpha + (1 - alpha) / 2]).to(
                    predictions.device
                ),
                dim=0,
            )
        else:
            point_estimate = predictions.mean(dim=0)
            std_of_pt_ests = predictions.std(dim=0)
            lo, hi = (
                point_estimate - 2 * std_of_pt_ests,
                point_estimate + 2 * std_of_pt_ests,
            )
        assert lo.shape == hi.shape == y_test.shape
        interval_scores_for_each_feature_for_each_test_case = (
            (hi - lo)
            + (2 / alpha) * (lo - y_test) * (y_test < lo)
            + (2 / alpha) * (y_test - hi) * (y_test > hi)
        )
        avgs_among_output_features = (
            interval_scores_for_each_feature_for_each_test_case.mean(dim=1)
        )
        return avgs_among_output_features  # ~~~ figure 8b in the SLOSH paper is a a box+whisker plot of these n_test values


#
# ~~~ Compute energy scores
def energy_scores(predictions, y_test):
    with torch.no_grad():
        n_test, n_out_features = y_test.shape
        es = []
        #
        # ~~~ Apply equation (14) of the SLOSH paper to each of the n_test test cases
        for j, Y in enumerate(y_test):
            Y_tilde = predictions[
                :, j, :
            ]  # ~~~ has shape (N_POSTERIOR_SAMPLES, n_out_features)
            Y = Y.reshape(1, n_out_features)
            es.append(
                torch.cdist(Y_tilde, Y).mean()
                - (1 / 2) * torch.cdist(Y_tilde, Y_tilde).mean()
            )
        return torch.stack(es)


#
# ~~~ Compute the interval score for all predictions "averaged over the spatial domain" (dim=1) instead of over the testing dataset (dim=0)
def aggregate_covarge(predictions, y_test, quantile_uncertainty, alpha=0.95):
    with torch.no_grad():
        if quantile_uncertainty:
            # fmt: off
            lo, hi = predictions.quantile(
                    q   = torch.Tensor([ (1-alpha)/2, alpha+(1-alpha)/2 ]).to(predictions.device),
                    dim = 0
                )
            # fmt: on
        else:
            point_estimate = predictions.mean(dim=0)
            std_of_pt_ests = predictions.std(dim=0)
            lo, hi = (
                point_estimate - 2 * std_of_pt_ests,
                point_estimate + 2 * std_of_pt_ests,
            )
        assert lo.shape == hi.shape == y_test.shape
        true_data_is_in_confidence_interval = (lo < y_test) * (y_test < hi)
        return true_data_is_in_confidence_interval.float().mean().item()
