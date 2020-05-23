

# compute regularization hyperparameter
def get_lambda(lambda0, lambda_schedule, n_total_iter):
    s = lambda_schedule
    if s == 0:
        return lambda0
    else:
        return lambda0 * float(min(n_total_iter, s)) / s


def creat_vis_plot(viz, x_value, y_value,
                   x_label, y_label, title, legend):
    win = viz.line(
        X=x_value,
        Y=y_value,
        opts=dict(
            xlabel=x_label,
            ylabel=y_label,
            title=title,
            legend=legend
        )
    )
    return win


def update_vis(viz, window, x_value, y_value):
    viz.line(
        X=x_value,
        Y=y_value,
        win=window,
        update='append'
    )
