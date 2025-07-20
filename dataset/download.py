import torchaudio
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

data = torchaudio.datasets.VCTK_092('./', download=True)

# for url in [
#             "dev-clean",
#             "dev-other",
#             "test-clean",
#             "test-other",
#             "train-clean-100",
#             "train-clean-360",
#             "train-other-500",
#         ]:
#     data = torchaudio.datasets.LIBRITTS('./', download=True)