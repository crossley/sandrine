library(data.table)
library(ggplot2)

rm( list = ls() )

phase_names_f <- c('Prebaseline_GazeData', 'Familiarisation_GazeData',
                   'Baseline_NFB_GazeData', 'Baseline_GazeData',
                   'Postbaseline_GazeData', 'Training_GazeData',
                   'Generalisation_GazeData', 'Relearning_GazeData',
                   'Washout_GazeData')

phase_names <- c('Prebaseline', 'Familiarisation', 'Baseline_NFB', 'Baseline_S',
                 'Postbaseline', 'Training', 'Generalisation', 'Relearning',
                 'Washout')

phase_order <- 1:9

ldf_rec <- list()
for (i in 1:length(phase_names_f)) {
    f_names <- list.files(paste('../data', sep=''),
                          pattern=paste(phase_names_f[i]),
                          full.names=TRUE)
    ldf <- lapply(seq_along(f_names),
                  function(j) {
                      z<-fread(f_names[j]);
                      setnames(z, c('trial', 'target', 'x', 'y', 't'));
                      })
    ldf <- lapply(seq_along(ldf), function(z) ldf[[z]][, sub := rep(z, .N)])
    ldf <- lapply(seq_along(ldf), function(z) ldf[[z]][, phase := rep(phase_names[i], .N)])
    ldf <- lapply(seq_along(ldf), function(z) ldf[[z]][, phase_order := rep(phase_order[i], .N)])
    ldf_rec <- c(ldf_rec, ldf)
}

d <- rbindlist(ldf_rec)

## spline trajectories
spline_func <- function(z,n) {
    x <- z[, x]
    y <- z[, y]
    t <- z[, t_trial]

    xs <- spline(t, x, n=n, method="fmm")
    ys <- spline(t, y, n=n, method="fmm")

    ts <- as.numeric(unlist(xs[1]))
    xs <- as.numeric(unlist(xs[2]))
    ys <- as.numeric(unlist(ys[2]))

    d_spline = data.table(t=ts, x=xs, y=ys)
    return(d_spline)
}

d[, t_trial := t - min(t), .(sub, trial, target, phase)]

dd <- d[, spline_func(.SD, 100), .(t_trial, phase, sub, target), .SDcols=c('x', 'y', 't_trial')]

ddd <- dd[sub==3, .(mean(x), mean(y)), .(phase, target, t_trial)]
ggplot(ddd[t_trial < 0.4], aes(V1, V2)) +
    geom_point(alpha=0.1) +
    xlim(-0.3, 0.3) +
    ylim(0.1, 0.3) +
    facet_wrap(~phase)
