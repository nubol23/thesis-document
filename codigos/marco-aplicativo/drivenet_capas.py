base_net = mobilenet_v2(pretrained=pretrained)

self.features = base_net.features
self.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 251),
)
self.concat_fc = nn.Linear(256, num_classes)
