Pod::Spec.new do |s|
  s.name           = 'Camerax'
  s.version        = '1.0.0'
  s.summary        = 'A sample project summary'
  s.description    = 'A sample project description'
  s.author         = ''
  s.homepage       = 'https://docs.expo.dev/modules/'
  s.platforms      = {
    :ios => '15.1',
    :tvos => '15.1'
  }
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'
  s.dependency 'TensorFlowLiteSwift', '~> 2.14.0'
  s.dependency 'TensorFlowLiteSwift/CoreML', '~> 2.14.0'
  s.dependency 'TensorFlowLiteSwift/Metal', '~> 2.14.0'
  s.dependency 'MediaPipeTasksVision'

  # Swift/Objective-C compatibility
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
  }

  s.source_files = "**/*.{h,m,mm,swift,hpp,cpp}"
  s.resources = "assets/*.{tflite,task}"
end
