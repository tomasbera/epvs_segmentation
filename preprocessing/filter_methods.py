import SimpleITK as sitk


def N4_bias_filter(raw_image, shrink_factor=4):
    raw_image = sitk.Cast(raw_image, sitk.sitkFloat32)

    mask_image = sitk.LiThreshold(raw_image, 0, 1)

    input_image = sitk.Shrink(raw_image, [shrink_factor] * raw_image.GetDimension())
    mask_image_small = sitk.Shrink(mask_image, [shrink_factor] * mask_image.GetDimension())

    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_shrunk = bias_corrector.Execute(input_image, mask_image_small)

    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_image)
    corrected_full = raw_image / sitk.Exp(log_bias_field)

    return corrected_full
