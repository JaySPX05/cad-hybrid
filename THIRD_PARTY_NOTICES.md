# Third-party notices

This project uses the third-party software and pretrained models listed below. Their licences require the notices in this file to be kept with any distribution of this project, for example a packaged app or shipped model files.

Datasets are credited separately in [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md).

---

## TabICL

- **What we use:** the `tabicl` Python package (PyPI) and the pretrained classifier checkpoint `tabicl-classifier-v2-20260212.ckpt`. It is downloaded from Hugging Face (`jingang/TabICL`) on first use and serves as the deep tabular model in the lab branch.
- **Source:** https://github.com/soda-inria/tabicl
- **Licence:** BSD 3-Clause. The licence text below is copied from the repository's `LICENSE` file on 24 Oct 2026.
- **Note:** the repository states that code in its `src/tabicl/forecast` directory is derived from TabPFN-TS and licensed separately under Apache 2.0. We don't use the forecasting code.

```
BSD 3-Clause License

Copyright (c) 2025, Soda team @ Inria

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
