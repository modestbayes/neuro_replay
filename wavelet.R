library(wavelets)

# wavelet transform of LFP data per trial
# 256 time steps x 15 channels multivariate series
# as levels increase, lower frequencies and fewer coefficients
lfp_current <- as.numeric(lfp_training_matrix[1, ])
lfp_multi <- t(matrix(data=lfp_current, nrow=15, ncol=256, byrow=TRUE))
output <- dwt(lfp_multi)
wav_coef <- output@W
lfp_features <- c(wav_coef[9]$W, wav_coef[5]$W5, wav_coef[4]$W4)

# loop through all the trials and extract features
all_lfp_features <- NULL
for (i in 1:248) {
  lfp_current <- as.numeric(lfp_training_matrix[i, ])
  lfp_multi <- t(matrix(data=lfp_current, nrow=15, ncol=512, byrow=TRUE)[, 1:256])
  stuff <- dwt(lfp_multi, n.levels=6)
  wav_coef <- stuff@W
  lfp_features <- c(wav_coef[6]$W6, wav_coef[5]$W5, wav_coef[4]$W4)
  all_lfp_features <- rbind(all_lfp_features, lfp_features)
}


# double check by univariate wavelet transform
lfp_features <- NULL
for (j in 1:15) {
  lfp_uni <- lfp_multi[, j]
  output <- dwt(lfp_uni)
  wav_coef <- output@W
  lfp_features <- c(lfp_features, wav_coef[5]$W5, wav_coef[4]$W4)
}


all_lfp_features <- NULL
for (i in 1:248) {
  lfp_current <- as.numeric(lfp_training_matrix[i, ])
  lfp_multi <- t(matrix(data=lfp_current, nrow=15, ncol=256, byrow=TRUE))
  lfp_features <- NULL
  for (j in 1:15) {
    lfp_uni <- lfp_multi[, j]
    output <- dwt(lfp_uni)
    wav_coef <- output@W
    lfp_features <- c(lfp_features, wav_coef[6]$W6, wav_coef[5]$W5, wav_coef[4]$W4)
  }
  all_lfp_features <- rbind(all_lfp_features, lfp_features)
}

write.table(all_lfp_features, file='all_lfp_features.csv', row.names=FALSE, col.names=FALSE, sep=',')


library(wavethresh)
coeffs <- wd(lfp_multi[, 1], family="DaubExPhase", type="station")
accessD(coeffs, 2)