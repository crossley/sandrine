library(data.table)
library(ggplot2)

rm( list = ls() )

col_names <- c("Target", "Trial_Phase", "Appl_Perturb",
               "imv_X", "imv_Y",
               "Endpoint_X", "Endpoint_Y",
               "imv_Error", "Endpoint_Error",
               "imv_Error_Mean", "Endpoint_Error_Mean",
               "MT", "RT", "Max_Vel")

phase_names_f <- c('Prebaseline_SubjData', 'Familiarisation_SubjData',
                 'Baseline_NFB_SubjData', 'Baseline_S_SubjData',
                 'Postbaseline_SubjData', 'Training_SubjData',
                 'Generalisation_SubjData', 'Relearning_SubjData',
                 'Washout_SubjData')

phase_names <- c('Prebaseline', 'Familiarisation', 'Baseline_NFB', 'Baseline_S',
                 'Postbaseline', 'Training', 'Generalisation', 'Relearning',
                 'Washout')

phase_order <- 1:9

ldf_rec <- list()
for (i in 1:length(phase_names_f)) {
    f_names <- list.files(paste('../data/', sep=''),
                          pattern=paste(phase_names_f[i]),
                          full.names=TRUE)

    ldf <- lapply(f_names, function(z) {z<-fread(z); setnames(z,col_names)})
    ldf <- lapply(seq_along(ldf), function(z) ldf[[z]][, sub := rep(z, .N)])
    ldf <- lapply(seq_along(ldf), function(z) ldf[[z]][, phase := rep(phase_names[i], .N)])
    ldf <- lapply(seq_along(ldf), function(z) ldf[[z]][, phase_order := rep(phase_order[i], .N)])
    ldf_rec <- c(ldf_rec, ldf)
}

d <- rbindlist(ldf_rec)
d <- d[order(sub)]

## NOTE: fix target goofiness
d[Target %in% 13:24, Target := as.integer(Target - 12)]

## NOTE: add target_deg indicator column
d[, target_deg := -1]
d[Target == 1, target_deg := 0]
d[Target == 2, target_deg := 30]
d[Target == 3, target_deg := 60]
d[Target == 4, target_deg := 90]
d[Target == 5, target_deg := 120]
d[Target == 6, target_deg := 150]
d[Target == 7, target_deg := 180]
d[Target == 8, target_deg := -150]
d[Target == 9, target_deg := -120]
d[Target == 10, target_deg := -90]
d[Target == 11, target_deg := -60]
d[Target == 12, target_deg := -30]

## NOTE: Try to fix hand angle madness
d[, HA := -atan2(Endpoint_Y - 0.2, Endpoint_X) * 180 / pi + 90]
d[Endpoint_Y - 0.2 < 0 & Endpoint_X < 0 & target_deg != 180, HA := HA - 360]
d[, HA := HA - target_deg]

## NOTE: inspect target locations
ggplot(d, aes(Endpoint_X, Endpoint_Y-0.2, color=as.factor(target_deg))) +
    geom_point() +
    facet_wrap(~phase)

ggplot(d, aes(target_deg, HA, color=as.factor(target_deg))) +
    geom_point() +
    facet_wrap(~phase)


## NOTE: quick hack switch
d[, Endpoint_Error := HA]

## NOTE: add trial indicator column
d <- d[order(sub, phase_order, Trial_Phase)]
d[, trial := 1:.N, .(sub)]

## NOTE: add aware indicator column
subs_explicit = c(1, 2, 3, 5, 9, 10, 16, 18, 19, 20, 24, 28, 29, 30, 31, 33, 34,
                  40, 41, 44, 45, 46, 47, 48)
d[, aware := 'implicit']
d[sub %in% subs_explicit, aware := 'explicit']

## NOTE: add cw / ccw indicator column
subs_rot <- c(1, 2, 2, 1, 2, 2, 0, 2, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 2,
              0, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 1, 0, 0, 2, 1, 2, 1, 2,
              2, 2, 2, 1)
condition <- rep(subs_rot, each=d[, .N, .(sub)][, unique(N)])
d[, cnd := condition]

## NOTE: add rot_dir indicator column
subs_ccw = c(1, 3, 4, 6, 7, 10, 12, 14, 15, 16, 17, 19, 21, 24, 29, 31, 33, 34,
             35, 36, 37, 41, 43, 47)
d[, rot_dir := 'cw']
d[sub %in% subs_ccw, rot_dir := 'ccw']

## NOTE: flip rot_dir so that everybody can be plotted in the same space
d[rot_dir == 'cw' &
  phase %in% c('Training', 'Generalisation', 'Relearning', 'Washout'),
  Endpoint_Error := -1 * Endpoint_Error]

## NOTE: Also flip the rotation for model fitting in python
d[rot_dir == 'cw' &
  phase %in% c('Training', 'Generalisation', 'Relearning', 'Washout'),
  Appl_Perturb := -1 * Appl_Perturb]

## NOTE: add phase means for baseline correction
d[, ee_mean := mean(Endpoint_Error, na.rm=T), .(sub, phase, Target)]
d[, ee_mean_correction_fb := ee_mean[which(phase=="Baseline_S")][1], .(Target, rot_dir)]
d[, ee_mean_correction_nfb := ee_mean[which(phase=="Baseline_NFB")][1], .(Target, rot_dir)]

## NOTE: Perform baseline correction
d[, bcee := Endpoint_Error]
d[phase %in% c("Generalisation", "Washout"), bcee := Endpoint_Error - ee_mean_correction_nfb] # ???
d[phase %in% c("Postbaseline", "Training", "Relearning"), bcee := Endpoint_Error - ee_mean_correction_nfb]

## NOTE: plot all trials collapsed over CW / CCW
dd <- d[, .(mean(Endpoint_Error, na.rm=T), sd(Endpoint_Error, na.rm=T)/sqrt(16)), .(cnd, trial)]
ggplot(dd, aes(trial, V1, colour=as.factor(cnd))) +
    geom_point(alpha=0.4)
ggsave('../figures/learning_curves.pdf', width=10, height=4)

## NOTE: plot learning curves per cnd
dd <- d[, .(mean(Endpoint_Error, na.rm=T), sd(Endpoint_Error, na.rm=T)/sqrt(16)), .(cnd, trial)]
ggplot(dd, aes(trial, V1, colour=as.factor(cnd))) +
    geom_point(alpha=1.0) +
    facet_wrap(~cnd)
ggsave('../figures/learning_curves_percnd.pdf', width=10, height=4)


# NOTE: Reaction time
dd <- d[, .(mean(RT, na.rm=T), sd(RT, na.rm=T)/sqrt(16)), .(cnd, trial, rot_dir)]
ggplot(dd, aes(trial, V1, colour=as.factor(cnd))) +
    geom_point(alpha=0.4) +
    facet_wrap(~rot_dir)
ggsave('../figures/learning_curves_cwccw_RT.pdf', width=10, height=4)


## NOTE: plot all trials expanded over CW / CCW
dd <- d[, .(mean(Endpoint_Error, na.rm=T), sd(Endpoint_Error, na.rm=T)/sqrt(16)), .(cnd, trial, rot_dir)]
ggplot(dd, aes(trial, V1, colour=as.factor(cnd))) +
    geom_point(alpha=0.4) +
    facet_wrap(~rot_dir)
ggsave('../figures/learning_curves_cwccw.pdf', width=10, height=4)

dd <- d[, .(mean(HA, na.rm=T), sd(HA, na.rm=T)/sqrt(16)), .(cnd, trial, rot_dir)]
ggplot(dd, aes(trial, V1, colour=as.factor(cnd))) +
    geom_point(alpha=0.4) +
    facet_wrap(~rot_dir)
ggsave('../figures/learning_curves_cwccw_HA.pdf', width=10, height=4)

## NOTE: plot all trials separate over CW / CCW
dd <- d[, mean(Endpoint_Error, na.rm=T), .(cnd, trial, rot_dir, aware)]
ggplot(dd, aes(trial, V1, colour=as.factor(cnd))) +
    geom_point(alpha=0.1) +
    facet_wrap(~rot_dir*aware)

## NOTE: take a look at prebaseline histograms
dd <- d[phase == 'Prebaseline',
        mean(Endpoint_Error, na.rm=T),
        .(cnd, trial, rot_dir)]
ggplot(dd, aes(V1, fill=as.factor(cnd))) +
    geom_histogram(alpha=0.75) +
    facet_wrap(~rot_dir*cnd)
ggsave('../figures/prebaseline.pdf', width=8, height=6)


## NOTE: look at prebaseline bias per target
dd <- d[phase == 'Prebaseline',
        .(mean(Endpoint_Error, na.rm=T), sd(Endpoint_Error, na.rm=T)/sqrt(16)),
        .(cnd, rot_dir, target_deg)]
ggplot(dd, aes(target_deg, V1, colour=as.factor(cnd))) +
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin=V1-V2, ymax=V1+V2)) +
    facet_wrap(~rot_dir*cnd)
ggsave('../figures/prebaseline_bias.pdf', width=8, height=6)


## NOTE: Plot generalisation function --- no baseline correction
dd <- d[phase == "Generalisation",
        .(mean(Endpoint_Error, na.rm = T), sd(Endpoint_Error, na.rm = T)/sqrt(.N)),
        .(cnd, target_deg, rot_dir)]

ggplot(dd, aes(x = target_deg, y = V1, colour = as.factor(cnd))) +
    geom_line() +
    geom_errorbar(aes(ymin=V1-V2, ymax=V1+V2), width=10) +
    scale_x_continuous(breaks=c(0,30,60,90,120,150,-150,-120,-90,-60,-30)) +
    xlim(-150, 150) +
    facet_wrap(~rot_dir)
ggsave('../figures/generalisation_ee.pdf', width=10, height=4)


## NOTE: Plot generalisation function --- baseline correction
dd <- d[phase == "Generalisation",
        .(mean(bcee, na.rm = T), sd(bcee, na.rm = T)/sqrt(16)),
        .(cnd, target_deg, rot_dir)]

ggplot(dd, aes(x = target_deg, y = V1, colour = as.factor(cnd))) +
    geom_line() +
    geom_errorbar(aes(ymin=V1-V2, ymax=V1+V2), width=10) +
    scale_x_continuous(breaks=c(0,30,60,90,120,150,-150,-120,-90,-60,-30)) +
    xlim(-150, 150) +
    facet_wrap(~rot_dir)
ggsave('../figures/generalisation_bcee.pdf', width=10, height=4)


## NOTE: Plot generalisation function --- baseline correction --- % adaptation
d[, adapt_level := mean(bcee[which(phase == 'Training')], na.rm=T), .(cnd, rot_dir)]

dd <- d[phase == "Generalisation",
        .(mean(bcee / adapt_level, na.rm = T), sd(bcee / adapt_level, na.rm = T)/sqrt(16)),
        .(cnd, target_deg, rot_dir)]

ggplot(dd, aes(x = target_deg, y = V1, colour = as.factor(cnd))) +
    geom_line() +
    geom_errorbar(aes(ymin=V1-V2, ymax=V1+V2), width=10) +
    scale_x_continuous(breaks=c(0,30,60,90,120,150,-150,-120,-90,-60,-30)) +
    xlim(-150, 150) +
    ylim(-0.2, 1.0) +
    facet_wrap(~rot_dir)
ggsave('../figures/generalisation_bcee_percent_adapt.pdf', width=10, height=4)



## NOTE: Does "aware" matter?
dd <- d[phase == "Generalisation",
        .(mean(bcee, na.rm = T), sd(bcee, na.rm = T)/sqrt(.N)),
        .(cnd, target_deg, rot_dir, aware)]

ggplot(dd, aes(x = target_deg, y = V1, colour = as.factor(cnd))) +
    geom_line() +
    geom_errorbar(aes(ymin=V1-V2, ymax=V1+V2), width=10) +
    scale_x_continuous(breaks=c(0,30,60,90,120,150,-150,-120,-90,-60,-30)) +
    xlim(-150, 150) +
    facet_wrap(~aware*rot_dir)
ggsave('../figures/generalisation_bcee_aware.pdf', width=10, height=8)


## NOTE: Collapse across cw / ccw
dd <- d[phase == "Generalisation",
        .(mean(bcee, na.rm = T), sd(bcee, na.rm = T)/sqrt(.N)),
        .(cnd, target_deg)]

ggplot(dd, aes(x = target_deg, y = V1, colour = as.factor(cnd))) +
    geom_line() +
    geom_errorbar(aes(ymin=V1-V2, ymax=V1+V2), width=10) +
    scale_x_continuous(breaks=c(0,30,60,90,120,150,-150,-120,-90,-60,-30)) +
    xlim(-150, 150)
ggsave('../figures/generalisation_bcee_collapsed.pdf', width=10, height=8)


## NOTE: Collapse across cnd
dd <- d[phase == "Generalisation",
        .(mean(Endpoint_Error, na.rm = T), sd(bcee, na.rm = T)/sqrt(.N)),
        .(rot_dir, target_deg)]

ggplot(dd, aes(x = target_deg, y = V1, colour = as.factor(rot_dir))) +
    geom_line() +
    geom_errorbar(aes(ymin=V1-V2, ymax=V1+V2), width=10) +
    scale_x_continuous(breaks=c(0,30,60,90,120,150,-150,-120,-90,-60,-30)) +
    xlim(-150, 150)
ggsave('../figures/generalisation_bcee_collapsed_rot_dir.pdf', width=10, height=8)



## NOTE: write master data.table to a csv for reading into Python
d[rot_dir == 'cw' &
  phase %in% c('Training', 'Generalisation', 'Relearning', 'Washout'),
  Appl_Perturb := -1 * Appl_Perturb]

d[rot_dir == 'cw' &
  phase %in% c('Training', 'Generalisation', 'Relearning', 'Washout'),
  Endpoint_Error := -1 * Endpoint_Error]

d[rot_dir == 'cw' &
  phase %in% c('Training', 'Generalisation', 'Relearning', 'Washout'),
  bcee := -1 * bcee]

fwrite(d, '../fit_input/master_data.csv')
