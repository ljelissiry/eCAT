# Third-Party Notices

eCAT is licensed under the MIT License. See `LICENSE` for the eCAT license text.

eCAT uses several open-source Python packages for data handling, plotting,
analysis, the optional Workbench app, and optional simulation workflows. The
table below summarizes eCAT's direct dependencies, and the ElectroKitty section
retains the full license notice for the optional simulation backend.

Complete dependency license details are available from each upstream package.

## Direct Dependencies

| Package | Role in eCAT | License / notice summary |
|---|---|---|
| IPython | Notebook and interactive display support | BSD 3-Clause |
| Jinja2 | Template rendering | BSD 3-Clause |
| Matplotlib | Plotting and figure export | Matplotlib License, BSD-compatible |
| NumPy | Numerical arrays and calculations | BSD 3-Clause, with additional bundled notices in upstream metadata |
| openpyxl | Excel workbook reading and writing | MIT License |
| pandas | DataFrame and tabular data handling | BSD 3-Clause |
| scikit-learn | Fitting/model utility support | BSD 3-Clause |
| SciPy | Scientific fitting and numerical routines | BSD 3-Clause, with additional bundled notices in upstream metadata |
| Dash | Optional eCAT Workbench app UI | MIT License |
| dash-ag-grid | Optional eCAT Workbench table UI | MIT License |
| pywebview | Optional local app window for eCAT Workbench | BSD 3-Clause |
| ElectroKitty | Optional CV simulation backend | BSD 3-Clause; full notice retained below |

Dependency versions and complete license files are provided by the installed
Python packages. For ordinary source installs, these packages are installed as
external dependencies rather than vendored source code.

## ElectroKitty

eCAT can use ElectroKitty as an optional backend for CV simulation and fitting.
ElectroKitty is developed by Ožbej Vodeb and is licensed under the BSD 3-Clause
License.

The license text below was copied from the ElectroKitty GitHub repository
(`RedrumKid/ElectroKitty`) at tag `Release`, commit
`d1c5f37b442321f8b5bcf48fd9fd76cdd69daef4`, corresponding to ElectroKitty
package version `1.0.11.5`.

```text
BSD 3-Clause License

Copyright (c) 2024, Ožbej Vodeb

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
