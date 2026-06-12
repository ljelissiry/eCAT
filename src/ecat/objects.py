"""Core eCAT object classes.

This module exposes the object classes used by the public eCAT API.
"""

from .utils import *  # noqa: F401,F403
from .options import *  # noqa: F401,F403
from .parsers import exp_type_short as _exp_type_short
from ._plot_style import _active_plot_style_value


def _integrate_trapezoid(y, x):
    integrator = getattr(np, "trapezoid", None)
    if integrator is None:
        integrator = np.trapz
    return integrator(y, x)


def _first_fit_color(options, fallback="tab:red"):
    value = (options or {}).get("fit colors")
    if value is None:
        value = (options or {}).get("fit color")
    if value is None:
        return fallback
    if not isinstance(value, str):
        try:
            if mpl.colors.is_color_like(value):
                return value
        except Exception:
            pass
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            values = list(value)
            return values[0] if values else fallback
    return value


def _symbol_labels_enabled(value):
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() == "auto" and bool(_active_plot_style_value("symbol labels"))


def _axis_label_symbol(axis_name):
    key = str(axis_name).strip().lower()
    key = key.replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key)
    base = key.split(" vs ", 1)[0].strip()
    if base.startswith("current density"):
        return "j"
    if base.startswith("potential"):
        return "E"
    if base.startswith("current"):
        return "i"
    if base in {"time", "t", "duration"} or base.startswith("time "):
        return "t"
    if base.startswith("charge"):
        return "Q"
    return None


def _symbolized_axis_name(axis_name):
    symbol = _axis_label_symbol(axis_name)
    if not symbol:
        return axis_name
    axis_name = str(axis_name)
    if " vs " in axis_name:
        _base, ref = axis_name.split(" vs ", 1)
        return f"${symbol}$ vs {ref}"
    return f"${symbol}$"


class ChronoAnalysisResult(dict):
    """Dictionary-compatible container for CA/CP analysis outputs."""

    def __init__(self, values=None, *, axes=None):
        super().__init__({} if values is None else values)
        self.axes = axes


class CVAnalysisResult(dict):
    """Dictionary-compatible container for single-CV analysis outputs."""

    def __init__(
        self,
        values=None,
        *,
        primary=None,
        table=None,
        summary=None,
        diagnostics=None,
        figure=None,
        axes=None,
    ):
        super().__init__({} if values is None else values)
        self.primary = primary
        self.table = table if table is not None else pd.DataFrame(columns=["Metric", "Value"])
        self.summary = {} if summary is None else summary
        self.diagnostics = {} if diagnostics is None else diagnostics
        self.figure = figure
        self.axes = axes

    def show(self, options=None):
        """Display or print the human-readable analysis table."""
        options = {} if options is None else dict(options)
        pretty_print = bool(options.get("pretty print", True))
        header = options.get("header", True)
        if header is True:
            header = _cv_analysis_title(self.summary.get("analysis"))
        if pretty_print and display is not None:
            display_table = _cv_analysis_pretty_table(self.table)
            styled = (
                display_table.style
                .hide(axis="index")
                .format(escape=None)
                .set_properties(**{"text-align": "left"})
                .set_table_styles([
                    {"selector": "th", "props": [("text-align", "left")]},
                    {"selector": "td", "props": [("text-align", "left")]},
                ])
            )
            if header:
                styled = styled.set_caption(str(header))
            display(styled)
        else:
            if header:
                print(f"{header}:")
            print(self.table.to_string(index=False))
        return self.table


def _cv_analysis_title(analysis):
    if not analysis:
        return None
    labels = {
        "peak_potential": "Peak Potential",
        "peak_current": "Peak Current",
        "half_peak_potential": "Half-Peak Potential",
        "half_wave_potential": "Half-Wave Potential",
        "peak_info": "Peak Info",
        "wave_info": "Wave Info",
        "current_at_potential": "Current At Potential",
    }
    key = str(analysis)
    return labels.get(key, key.replace("_", " ").title())


def _cv_analysis_metric_label(metric):
    labels = {
        "Ep": "E<sub>p</sub>",
        "ip": "i<sub>p</sub>",
        "Ep/2": "E<sub>p/2</sub>",
        "Δ(Ep - Ep/2)": "Δ(E<sub>p</sub> - E<sub>p/2</sub>)",
        "E(1/2)": "E<sub>1/2</sub>",
        "P1 Ep": "P1 E<sub>p</sub>",
        "P1 ip": "P1 i<sub>p</sub>",
        "P2 Ep": "P2 E<sub>p</sub>",
        "P2 ip": "P2 i<sub>p</sub>",
    }
    return labels.get(str(metric), metric)


def _cv_analysis_pretty_table(table):
    display_table = table.copy()
    if "Metric" in display_table.columns:
        display_table["Metric"] = display_table["Metric"].map(_cv_analysis_metric_label)
    return display_table


def _cv_analysis_unit_text(unit):
    if unit is None:
        return ""
    unit = str(unit)
    if unit.startswith("u"):
        unit = "μ" + unit[1:]
    return unit


def _cv_analysis_format_number(value, sig_figs):
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return str(numeric)
    return f"{round_sigfigs(numeric, sig_figs):g}"


def _cv_analysis_display_value(cv_obj, value, row, options):
    kind = row.get("kind")
    sig_figs = options.get("sig figs", 4)
    unit = row.get("unit")
    scale = 1.0

    if kind == "potential":
        x = cv_obj.x(options)
        x_name = x.name
        unit = cv_obj.units.get(x_name, unit or "V")
        scale, unit = cv_obj.scale_axis(
            np.asarray([value], dtype=float),
            x_name,
            unit,
            options.get("x unit", "auto"),
        )
    elif kind == "current":
        y = cv_obj.y(options)
        y_name = y.name
        unit = cv_obj.units.get(y_name, unit or "A")
        scale, unit = cv_obj.scale_axis(
            np.asarray([value], dtype=float),
            y_name,
            unit,
            options.get("y unit", "auto"),
        )
    elif kind == "plain":
        unit = row.get("unit", unit)

    try:
        display_value = float(value) * float(scale)
    except (TypeError, ValueError):
        display_value = value

    text = _cv_analysis_format_number(display_value, sig_figs)
    unit = _cv_analysis_unit_text(unit)
    return f"{text} {unit}".strip() if unit else text


def _cv_analysis_table(cv_obj, rows, options):
    rows = list(rows or [])
    segment_values = [
        str(row.get("segment"))
        for row in rows
        if row.get("segment") not in (None, "")
    ]
    include_segment = len(set(segment_values)) > 1
    display_rows = []
    for row in rows:
        display_row = {"Metric": row.get("metric", "")}
        if include_segment:
            display_row["Segment"] = "" if row.get("segment") is None else str(row.get("segment"))
        display_row["Value"] = _cv_analysis_display_value(cv_obj, row.get("value"), row, options)
        display_rows.append(display_row)
    columns = ["Metric", "Segment", "Value"] if include_segment else ["Metric", "Value"]
    return pd.DataFrame(display_rows, columns=columns)


def _cv_analysis_result(
    cv_obj,
    analysis,
    values,
    rows,
    primary_key,
    options,
    diagnostics=None,
    summary=None,
):
    summary_data = {"analysis": analysis}
    if summary:
        summary_data.update(summary)
    return CVAnalysisResult(
        values,
        primary=values.get(primary_key),
        table=_cv_analysis_table(cv_obj, rows, options),
        summary=summary_data,
        diagnostics={} if diagnostics is None else diagnostics,
        axes=plt.gca() if plt.get_fignums() else None,
    )


class echem:
    """Base electrochemistry object for imported time-series experiments.
    
    Parameters
    ----------
    filepath : str or path-like, optional
        Text file to parse.
    options : dict or ImportOptions, optional
        Import and metadata options. See ``e.describe_options("get_data")``.
    
    Examples
    --------
    >>> obj = e.echem.from_file(path, {"software": "CH"})
    """

    @staticmethod
    def _read_header_lines(filepath, num_lines=60, encoding='ISO-8859-1'):
        """
        Read up to `num_lines` header lines from a text file.
        Returns a list of stripped lines.
        """
        if filepath is None or not str(filepath).endswith(".txt"):
            return []

        header_lines = []
        with open(filepath, "r", encoding=encoding) as f:
            for _ in range(num_lines):
                try:
                    header_lines.append(next(f).strip())
                except StopIteration:
                    break
        return header_lines

    @classmethod
    def detect_software(cls, filepath, options=None):
        """
        Detect the instrument software from the file header, unless explicitly
        provided in options.
        """
        options = {} if options is None else options

        software = options.get("software")
        if software is not None:
            return software

        header_lines = cls._read_header_lines(filepath, num_lines=10)
        header = "\n".join(header_lines)

        if "EC-Lab" in header:
            return "EC-Lab"
        elif any("Instrument Model" in line and "CH" in line for line in header_lines):
            return "CH"
        elif any("Experiment Type" in line for line in header_lines):
            return "BASI"

        return None

    @classmethod
    def detect_experiment_type(cls, filepath, options=None):
        """
        Detect experiment type without constructing a fully loaded object.

        Returns one of the strings used in subclass_map, or "Unknown".
        """
        options = {} if options is None else options

        # Optional manual override for unusual/custom workflows
        explicit_type = options.get("experiment type")
        if explicit_type is not None:
            return explicit_type

        # For custom readers, auto-promotion is not guaranteed unless the user
        # supplies 'experiment type' explicitly.
        if callable(options.get("custom reader")):
            return "Unknown"

        software = cls.detect_software(filepath, options)
        header_lines = cls._read_header_lines(filepath, num_lines=60)

        if software == "CH":
            if len(header_lines) >= 2:
                return header_lines[1].strip()

        elif software == "BASI":
            for line in header_lines:
                if "Experiment Type" in line:
                    return line.split(":", 1)[1].strip()

        elif software == "EC-Lab":
            if len(header_lines) >= 4:
                return header_lines[3].strip()

        return "Unknown"

    @classmethod
    def from_file(cls, filepath, options=None):
        """Load one electrochemistry file and return the appropriate eCAT object subclass.
        
        Parameters
        ----------
        filepath : str or path-like
            File to parse.
        options : dict or ImportOptions, optional
            Import and parser options. See ``e.describe_options("get_data")``.
        
        Returns
        -------
        echem
            Parsed echem, cv, dpv, ca, cp, or CPE object.
        
        Examples
        --------
        >>> obj = e.echem.from_file(path, {"software": "CH"})
        """
        options = {} if options is None else options

        # If called from a subclass, just construct that subclass directly.
        if cls is not echem:
            obj = cls(filepath, options)
            if options.get("print", False):
                from . import plotting
                plotting.show(obj, options)
            return obj

        exp_type = cls.detect_experiment_type(filepath, options)

        subclass_map = {
            "Cyclic Voltammetry": cv,
            "Chronopotentiometry": cp,
            "Galvanostatic Cycling with Potential Limitation": cp,
            "Amperometric i-t Curve": ca,
            "Differential Pulse Voltammetry": dpv,
            # Add more as needed
        }

        promoted_cls = subclass_map.get(exp_type, echem)
        obj = promoted_cls(filepath, options)
        if options.get("print", False):
            from . import plotting
            plotting.show(obj, options)
        return obj

    def __new__(cls, filepath=None, options=None):
        """
        Keep __new__ simple: only allocate the object.
        """
        return super().__new__(cls)

    def __init__(self, filepath=None, options=None):
        self.filepath = filepath
        self.options = import_options_to_legacy_dict(options)
        
        # Timing
        self.timestamp = getattr(self, "timestamp", None)
        try:
            self.creation_time, self.modification_time = get_file_times(filepath)
        except OSError:
            self.creation_time = None
            self.modification_time = None

        # Base/default attributes
        self.name = getattr(self, "name", "Unnamed EChem Object")
        self.data = getattr(self, "data", pd.DataFrame())
        self.type = getattr(self, "type", None)
        self.software = getattr(self, "software", self.options.get("software"))
        self.num_x_cols = getattr(self, "num_x_cols", 1)
        self.temperature = self.options.get('temperature', 298)
        self.electrode_area = self.options.get('electrode area', 0)
        self.delta_x = getattr(self, "delta_x", None)
        self.units = getattr(self, "units", {})
        self.segments = getattr(self, "segments", 1)
        self.gas = getattr(self, "gas", self.options.get('gas'))
        self.solvent = getattr(self, "solvent", self.options.get('solvent'))
        self.ir_comp_resistance = getattr(self, "ir_comp_resistance", None)
        self.ir_uncomp_resistance = getattr(self, "ir_uncomp_resistance", None)
        self.ir_comp_percent = getattr(self, "ir_comp_percent", None)

        # Reference / potential-shift metadata
        self.reference_shift = getattr(self, "reference_shift", None)
        self.reference_label = getattr(self, "reference_label", None)
        self.reference_mode = getattr(self, "reference_mode", "none")
        self.reference_source_file = getattr(self, "reference_source_file", None)
        self.reference_failure_message = getattr(self, "reference_failure_message", None)

        # Load data only if a filepath was provided
        if filepath is not None:

            self.data = self.read_file_data(filepath, self.options)

            self.name = os.path.basename(filepath[:filepath.rindex(".")])
            self.name = apply_text_alterations(
                self.name,
                self.options.get("name alterations")
            )

            self.get_data_from_name()
            self.modify_by_options(self.options)

    def get_data_from_name(self):
        temp_name = '_' + self.name + '_'

        gases = ['Ar', 'N2', 'CO', 'CO2']
        temp_name_lower = temp_name.lower()
        found_gases = [gas for gas in gases if f'_{gas.lower()}_' in temp_name_lower]
        if found_gases:
            self.gas = '/'.join(found_gases)

        solvents = ['H2O', 'THF', 'DME', 'MeCN', 'DCM', 'DMF', 'DMSO']
        for solvent in solvents:
            if f'_{solvent.lower()}_' in temp_name_lower:
                self.solvent = solvent

    def _parse_ir_compensation_from_lines(self, lines):
        def get_resistance(label):
            pattern = rf'{label}\s*\(ohm\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
            for line in lines:
                match = re.search(pattern, str(line), flags=re.IGNORECASE)
                if match:
                    return float(match.group(1))
            return None

        comp_r = get_resistance(r'Comp\s*R')
        uncomp_r = get_resistance(r'UC\s*R')
        if comp_r is not None and uncomp_r is None:
            uncomp_r = 0.0

        self.ir_comp_resistance = comp_r
        self.ir_uncomp_resistance = uncomp_r
        self.ir_comp_percent = None

        if comp_r is not None and uncomp_r is not None:
            total_r = comp_r + uncomp_r
            if total_r != 0:
                self.ir_comp_percent = 100 * comp_r / total_r

    def read_file_data(self, filepath, options=None):
        """
        Reads electrochemical data from the specified file.

        Parameters:
            filepath (str): The path to the file containing the electrochemical data.
            options (dict): A dictionary of options for data processing.

        Returns:
            pd.DataFrame: A DataFrame containing the electrochemical data.
        """
        if filepath is None:
            return pd.DataFrame()

        options = import_options_to_legacy_dict(options)

        self.software = options.get('software')
        if self.software is None:
            self.software = self.detect_software(filepath, options)

        # Custom reader override
        if 'custom reader' in options and callable(options['custom reader']):
            return options['custom reader'](self, filepath, options)

        # Dispatch dictionary
        software_readers = {
            "EC-Lab": self.read_eclab_txt,
            "BASI": self.read_basi_txt,
            "CH": self.read_ch_txt,
        }

        if self.software in software_readers:
            return software_readers[self.software](filepath, options)

        # Fallback generic parser if no match
        return self.read_generic_txt(filepath, options)

    def infer_delimiter(self, sample_line):
        """
        Infers the delimiter used in a sample line of data.
        """
        if '\t' in sample_line:
            return '\t'
        elif ',' in sample_line:
            return ','
        else:
            return r'\s+' # NOTE: check if works

    def read_basi_txt(self, filepath, options):
        """
        Reads data from a BASI-format .txt file with flexible column and delimiter handling.
        """
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            lines = [line.strip() for line in f.readlines()]

        # Try to find the header/data split
        data_start_idx = next(
            (i for i, line in enumerate(lines) if line.startswith('Potential') or line.startswith('[Begin Data]')),
            None)
        if data_start_idx is None:
            raise ValueError("Could not find start of data in BASI file.")

        # Try to parse experiment type
        for line in lines[:data_start_idx]:
            if 'Experiment Type' in line:
                self.type = line.split(':', 1)[1].strip()
                break
        else:
            self.type = 'Unknown'

        # Infer delimiter from the first data line after the header
        data_sample_line = lines[data_start_idx + 1]
        delimiter = self.infer_delimiter(data_sample_line)

        df = pd.read_csv(filepath, sep=delimiter, skiprows=data_start_idx + 1, engine='python', header=None)
        df = df.dropna(how='all', axis=1).dropna(how='any').reset_index(drop=True)

        # Use first row as header
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

        # Convert columns to numeric and strip units
        updated_columns = {}
        for col in df.columns:
            col_str = str(col)
            if '/' in col_str:
                name, unit = map(str.strip, col_str.split('/', 1))
                self.units[name] = unit
                updated_columns[col] = name
            else:
                updated_columns[col] = col_str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.rename(columns=updated_columns, inplace=True)
        df = df.dropna(subset=[df.columns[0]]).reset_index(drop=True)
        self.delta_x = round_sigfigs(abs(df.iloc[1, 0] - df.iloc[0, 0]), 3)

        return df

    def read_ch_txt(self, filepath, options):
        """
        Reads CH Instruments .txt data file with auto-detected delimiter and column count.
        """
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            lines = [line.strip() for line in f.readlines()]
        self._parse_ir_compensation_from_lines(lines)

        # Extract and convert time
        if len(lines) >= 1:
            time_str = lines[0]  # e.g. "Aug. 27, 2023   16:05:21"
            self.timestamp = _parse_ch_timestamp(time_str)

        # Assign software type from second line
        if len(lines) >= 2:
            self.type = lines[1]
        else:
            self.type = 'Unknown'

        data_start_idx = next(
            (i for i, line in enumerate(lines) if line.startswith('Potential') or line.startswith('Time')), None)
        if data_start_idx is None:
            raise ValueError("Could not find start of data in CH Instruments file.")

        # Infer delimiter
        data_sample_line = lines[data_start_idx]
        #print("|"+data_sample_line+"|")
        delimiter = self.infer_delimiter(data_sample_line)

        df = pd.read_csv(filepath, sep=delimiter, skiprows=data_start_idx, engine='python', header=None)
        df = df.dropna(how='all', axis=1).dropna().reset_index(drop=True)

        # Use first row as header
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

        # Convert to numeric and strip units
        updated_columns = {}
        for col in df.columns:
            if '/' in col:
                name, unit = map(str.strip, col.split('/', 1))
                self.units[name] = unit
                updated_columns[col] = name
            else:
                updated_columns[col] = col.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.rename(columns=updated_columns, inplace=True)
        df = df.dropna(subset=[df.columns[0]]).reset_index(drop=True)
        self.delta_x = round_sigfigs(abs(df.iloc[1, 0] - df.iloc[0, 0]), 3)

        return df

    def read_eclab_txt(self, file_path, options):
        import pandas as pd

        # Read header lines
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()

        # Extract experiment type from line 4
        if len(lines) >= 4:
            self.type = lines[3].strip()
        else:
            self.type = "Unknown"

        # Find number of header lines
        skiprows = None
        for line in lines:
            if "Nb header lines" in line:
                try:
                    skiprows = int(line.split(":")[1].strip().split()[0])
                    break
                except Exception:
                    raise ValueError("Could not parse number of header lines in EC-Lab file.")

        if skiprows is None:
            raise ValueError("Header line count not found in EC-Lab file.")

        # Load the data
        df = pd.read_csv(file_path, sep='\t', skiprows=skiprows - 1, encoding='latin1')

        # Normalize column names to lowercase for matching
        normalized_cols = {col.strip().lower(): col for col in df.columns}

        voltage_col_key = next((key for key in normalized_cols if "ewe" in key and "/" in key), None)
        current_col_key = next((key for key in normalized_cols if "<i>" in key and "/" in key), None)

        if voltage_col_key is None or current_col_key is None:
            raise ValueError(
                f"Could not find expected voltage/current columns.\nAvailable columns: {list(df.columns)}"
            )

        voltage_col = normalized_cols[voltage_col_key]
        current_col = normalized_cols[current_col_key]

        # Extract and rename relevant columns using original names
        df = df[[voltage_col, current_col]]

        # Determine imported units
        voltage_unit = voltage_col.split("/")[-1].strip()
        current_unit = current_col.split("/")[-1].strip()

        # Apply conversion factors to standard units (V and A)
        current_conversion = get_conversion_factor(current_unit)
        voltage_conversion = get_conversion_factor(voltage_unit)

        df[current_col] = df[current_col] * current_conversion
        df[voltage_col] = df[voltage_col] * voltage_conversion

        # Update column names to include standard format
        df = df.rename(columns={
            voltage_col: "Potential",
            current_col: "Current"
        })

        self.units["Potential"] = "V"
        self.units["Current"] = "A"

        # Calculate delta_x (average voltage step)
        voltages = df["Potential"]
        if len(voltages) > 1:
            self.delta_x = abs(voltages.iloc[1] - voltages.iloc[0])

        return df

    def read_generic_txt(self, filepath, options):
        """
        Fallback reader for general .txt files with minimal assumptions.
        """
        def is_numeric(value):
            try:
                float(value)
                return True
            except (TypeError, ValueError):
                return False

        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Infer header and delimiter
        data_start_idx = 0
        while not any(char.isdigit() for char in lines[data_start_idx]):
            data_start_idx += 1

        data_sample_line = lines[data_start_idx]
        delimiter = self.infer_delimiter(data_sample_line)
        read_start_idx = data_start_idx

        if data_start_idx > 0:
            previous_line = lines[data_start_idx - 1]
            prev_has_alpha = any(char.isalpha() for char in previous_line)

            if delimiter == r'\s+':
                prev_tokens = previous_line.split()
                data_tokens = data_sample_line.split()
                looks_like_header = (
                    prev_has_alpha
                    and len(prev_tokens) == len(data_tokens)
                    and any("/" in token for token in prev_tokens)
                )
            else:
                looks_like_header = (
                    prev_has_alpha
                    and delimiter in previous_line
                )

            if looks_like_header:
                read_start_idx = data_start_idx - 1

        df = pd.read_csv(filepath, sep=delimiter, skiprows=read_start_idx, engine='python', header=None)
        df = df.dropna(how='all', axis=1).dropna().reset_index(drop=True)

        # Use first row as header if it contains any non-numeric data
        if not all(is_numeric(x) for x in df.iloc[0]):
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)

        # Clean and convert
        updated_columns = {}
        for col in df.columns:
            col_str = str(col)
            if '/' in col_str:
                name, unit = map(str.strip, col_str.split('/', 1))
                self.units[name] = unit
                updated_columns[col] = name
            else:
                updated_columns[col] = col_str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.rename(columns=updated_columns, inplace=True)
        df = df.dropna(subset=[df.columns[0]]).reset_index(drop=True)
        self.delta_x = round_sigfigs(abs(df.iloc[1, 0] - df.iloc[0, 0]), 3)
        self.type = "Unknown"

        return df

    def extract_compounds_and_concentrations(self, extra_compounds=None):
        """
        Extracts compounds and concentrations from the data.

        Parameters:
            extra_compounds (str or None): User-defined option for additional compounds extraction. Default is None.

        Returns:
            tuple: A tuple containing the compounds (list) and concentrations (list).
        """
        # Create a temporary name with '_' prefix and suffix to facilitate regular expression matching
        temp_name = '_' + self.name + '_'
        
        # Regular expression pattern to extract compounds and concentrations from the temporary name
        compound_pattern = (
            r'(\d+(?:\.\d+)?)\s*'  # numeric value
            r'([munμ]?M|L|%|equiv|x)\s*'  # unit (mM, μM, L, %, equiv, x …)
            r'(?=[\w(\[])'  # next char must be a word-char **or an open parenthesis or bracket**
            r'([\w()\[\]\-+,]+?)_'  # the compound itself, allowing for characters including: [ ] + ,
        )
        
        # Use regular expressions to find all occurrences of compounds and concentrations in the temporary name
        def valid_concentration_match(item):
            _value, unit, compound = item
            if unit == "m":
                return False
            return compound.strip("_").lower() not in {
                "v", "vs", "mv", "mvs", "m", "mm", "um", "μm", "nm"
            }

        combined = [
            item for item in re.findall(compound_pattern, temp_name, flags=re.IGNORECASE)
            if valid_concentration_match(item)
        ]

        def normalize_concentration_unit(unit):
            unit_text = str(unit)
            lower = unit_text.lower()
            if lower == "m":
                return "M"
            if lower == "mm":
                return "mM"
            if lower in ("um", "μm"):
                return "μM" if "μ" in unit_text else "uM"
            if lower == "nm":
                return "nM"
            if lower == "l":
                return "L"
            if lower == "%":
                return "%"
            if lower == "equiv":
                return "equiv"
            if lower == "x":
                return "x"
            return unit_text

        # Format the extracted compounds and concentrations into separate lists
        compounds = [compound[-1].strip("_") for compound in combined]
        concentrations = [
            f"{compound[0]} {normalize_concentration_unit(compound[1])}"
            for compound in combined
        ]

        # Additional support for fraction-style gas tokens like _0.1CO2_
        fraction_gas_pattern = r'_(0?\.\d+)(CO2|CO|N2|Ar)_'
        for frac_str, gas_name in re.findall(fraction_gas_pattern, temp_name, flags=re.IGNORECASE):
            gas_name = gas_name.upper() if gas_name.lower() != "ar" else "Ar"
            frac = float(frac_str)
            percent = 100 * frac

            # avoid duplicating something already captured as e.g. 10%CO2
            pair = (gas_name, f"{percent:g} %")
            if not any(c == pair[0] and conc == pair[1] for c, conc in zip(compounds, concentrations)):
                compounds.append(gas_name)
                concentrations.append(pair[1])

        # If extra_compounds is provided, check for additional compounds in the filename and add them to the list
        if extra_compounds is not None:
            for compound in extra_compounds:
                if compound in self.name and compound not in compounds:
                    compounds.append(compound)

        # Return the resulting compounds and concentrations as a tuple
        return compounds, concentrations

    def modify_by_options(self,options):

        # setup options
        options = import_options_to_legacy_dict(options)

        # Extract compounds and concentrations from options, and assign them to the object's attributes
        self.compounds, self.concentrations = self.extract_compounds_and_concentrations(options['compounds'])
        #if self.compounds == []:
        #    self.compounds = ''

        options["electrode area"] = resolve_electrode_area_option(options)
        
        self.set_temperature(options['temperature'])
        self.set_electrode_area(options['electrode area'])
        if options['convert current'] != False:
            self.current_to(options['convert current'])
        if options.get('invert current', False):
            self.invert_current()
        if options['shift potential']:
            self.potential_shift(options)
        self._sync_data_unit_attrs()

    def _sync_data_unit_attrs(self):
        if isinstance(getattr(self, "data", None), pd.DataFrame):
            self.data.attrs["units"] = dict(getattr(self, "units", {}) or {})
        return self

    def set_electrode_area(self,area):
        self.electrode_area = area

    def set_temperature(self,temp):
        self.temperature = temp
        
    def current_to(self,unit):
        if "Current/A" in self.data.columns[-1]:
            if unit == "mA":
                self.data.rename(columns = {self.data.columns[-1]:"Current/mA"}, inplace = True)
                self.data["Current/mA"] *= 1000
            if unit == "uA" or unit == "μA":
                self.data.rename(columns = {self.data.columns[-1]:"Current/μA"}, inplace = True)
                self.data["Current/μA"] *= 1000000
            
    def to_density(self):
        if "Current" in self.data.columns[-1]:
            unit = self.data.columns[-1].split('/')[1]
            text = "Current Density/"+unit+"/cm$^2$"
            self.data.rename(columns = {self.data.columns[-1]:text}, inplace = True)
            self.data[text] /= self.electrode_area
            
    def to_min(self):
        if "Time/sec" in self.data.columns[0]:
            self.data.rename(columns = {self.data.columns[0]:"Time/min"}, inplace = True)
            self.data["Time/min"] /= 60

    def has_reference_shift(self):
        """
        Return True when a potential reference shift is active and can be applied
        to a raw 'Potential' column.
        """
        return (
            self.reference_shift is not None
            and "Potential" in self.data.columns
        )

    def reference_axis_name(self):
        """
        Name of the active shifted potential axis.
        """
        label = self.reference_label or "reference"
        return f"Potential vs {label}"

    def _shifted_potential_series(self):
        """
        Return the shifted potential as a virtual Series, without storing it in
        self.data.
        """
        if not self.has_reference_shift():
            raise ValueError("No active reference shift is stored on this object.")

        shifted = self.data["Potential"].copy() - self.reference_shift
        shifted.name = self.reference_axis_name()

        # Keep units discoverable through existing code paths
        if "Potential" in self.units and shifted.name not in self.units:
            self.units[shifted.name] = self.units["Potential"]

        return shifted

    def _available_x_column_names(self):
        available = [str(c) for c in self.data.columns[:self.num_x_cols]]

        if self.has_reference_shift():
            ref_name = self.reference_axis_name()
            if ref_name not in available:
                available.append(ref_name)

        return available

    def potential_shift(self, options):
        """
        Store the reference shift as metadata rather than inserting a new column.
        """
        shift = options.get("shift guess")
        label = options.get("shift label", "Fc/Fc+")

        if "Potential" not in self.data.columns:
            return

        if shift is None:
            return

        # Ignore unresolved auto strings here; get_data should resolve first
        if isinstance(shift, str):
            if shift.lower() == "auto":
                return
            try:
                shift = float(shift)
            except ValueError:
                return

        self.reference_shift = float(shift)
        self.reference_label = label

        # Preserve unit lookup behavior for downstream plotting/helpers
        axis_name = self.reference_axis_name()
        if "Potential" in self.units and axis_name not in self.units:
            self.units[axis_name] = self.units["Potential"]

    def x(self, options={}):
        """Return the selected x-axis data series.
        
        Parameters
        ----------
        options : dict or PlotOptions, optional
            Axis, segment, derivative, and scaling options. See ``describe_options("plot")``.
        
        Returns
        -------
        pandas.Series
            Selected x-axis data.
        
        Examples
        --------
        >>> potential = cv_obj.x({"x axis": "potential"})
        """
        options = {} if options is None else dict(options)
        """
        Access the x-data (e.g., time, potential) from the dataset.

        Behavior with shifted potentials:
        - If a reference shift is active and no explicit x-axis is requested,
          the default one-column x output is the shifted potential.
        - If one_column=False, both the raw x column(s) and the virtual shifted
          potential column are returned, preserving the old mental model.
        - If x axis is explicitly 'Potential', the raw potential is returned.
        - If x axis is explicitly 'Potential vs <label>', the shifted potential
          is returned virtually even though it is not stored in self.data.
        """
        one_column = options.get("one column", True)
        column_index = options.get("x column index", -1)
        column_name = options.get("x axis")
        scale = options.get("x scale", 1)

        # Explicit axis request
        if column_name:
            requested = str(column_name).strip()

            # First honor real stored columns
            try:
                actual_col = get_column_name_case_insensitive(requested, self.data.columns)
                series = self.data[actual_col].copy()
                return series * scale if scale != 1 else series
            except Exception:
                pass

            # Then honor the virtual shifted potential axis
            if self.has_reference_shift():
                ref_axis = self.reference_axis_name()
                if requested.lower() == ref_axis.lower():
                    series = self._shifted_potential_series()
                    return series * scale if scale != 1 else series

            raise ValueError(
                f"No x column found matching '{requested}'. "
                f"Available x columns: {self._available_x_column_names()} | "
                f"All dataframe columns: {list(self.data.columns)}"
            ) from exc

        if self.num_x_cols <= 0 or self.num_x_cols > self.data.shape[1]:
            raise ValueError(
                f"Invalid number of x columns configured: num_x_cols={self.num_x_cols}. "
                f"Available dataframe columns: {list(self.data.columns)}"
            )

        # Start from the real x columns only
        x_data = self.data.iloc[:, :self.num_x_cols].copy()

        # Add the shifted potential as a virtual x column when available.
        # This preserves the old behavior where x column index = -1 gives the
        # shifted axis, while x column index = 0 still gives raw Potential.
        if self.has_reference_shift():
            shifted = self._shifted_potential_series()
            x_data = pd.concat([x_data, shifted], axis=1)

        if one_column:
            ncols = x_data.shape[1]
            resolved_index = column_index if column_index >= 0 else ncols + column_index

            if resolved_index < 0 or resolved_index >= ncols:
                raise ValueError(
                    f"x column index {column_index} is out of range. "
                    f"Available x columns: {list(x_data.columns)}"
                )

            series = x_data.iloc[:, resolved_index].copy()
            return series * scale if scale != 1 else series

        if scale != 1:
            x_data = x_data * scale

        return x_data

    def y(self, options={}):
        """Return the selected y-axis data series.
        
        Parameters
        ----------
        options : dict or PlotOptions, optional
            Axis, derivative, normalization, and scaling options. See ``describe_options("plot")``.
        
        Returns
        -------
        pandas.Series
            Selected y-axis data.
        
        Examples
        --------
        >>> current = cv_obj.y({"y axis": "current"})
        """
        options = {} if options is None else dict(options)
        """
        Access the y-data (e.g., current, charge) from the dataset.

        Options:
            one column (bool): Return a single column (last or selected). Default is True.
            y scale (float): Scale factor to apply to y-values. Default is 1 (no scaling).
            y column index (int): Index of y-column to return if one_column is True. Default is -1 (last).
            y axis (str): Name of y-column to return. Overrides index-based access.

        Returns:
            pd.Series or pd.DataFrame: Selected and optionally scaled y-data.
        """
        one_column = options.get("one column", True)
        scale = options.get("y scale", 1)
        column_index = options.get("y column index", -1)
        column_name = options.get("y axis")
        smooth = options.get("smooth", False)
        offset = options.get("offset", 0)

        if self.num_x_cols < 0 or self.num_x_cols >= self.data.shape[1]:
            raise ValueError("Invalid number of x columns configured.")

        # Allow full override by column name (even outside the y-slice)
        if column_name:  # override always allowed
            actual_col = get_column_name_case_insensitive(column_name, self.data.columns)
            y_data = self.data[actual_col].copy()
        elif one_column:
            y_data = self.data.iloc[:, column_index].copy()
        else:
            # Default y-slice starts after the x-columns
            y_data = self.data.iloc[:, self.num_x_cols:].copy()

        derivative = options.get('derivative',0)
        derivative = 0 if derivative is None else derivative
        if smooth or derivative != 0:
            if isinstance(y_data, pd.Series):
                y_smooth, _ = _savgol_apply(y_data.to_numpy(dtype=float), options, deriv=derivative)
                y_data = pd.Series(y_smooth, index=y_data.index, name=y_data.name)
            else:
                y_data = y_data.apply(
                    lambda col: pd.Series(
                        _savgol_apply(col.to_numpy(dtype=float), options, deriv=derivative)[0],
                        index=col.index,
                        name=col.name,
                    ),
                    axis=0,
                )
        if scale != 1:
            y_data *= scale

        return y_data

    def xy(self, options={}):
        """Return selected x and y data series using one options dictionary.
        
        Parameters
        ----------
        options : dict or PlotOptions, optional
            Axis, segment, derivative, normalization, and scaling options. See ``describe_options("plot")``.
        
        Returns
        -------
        tuple of pandas.Series
            The selected x and y data.
        
        Examples
        --------
        >>> x, y = cv_obj.xy({"segment": 1})
        """
        return self.x(options), self.y(options)

    def xy_scale(self, options={}):
        x = self.x(options)
        x_name = x.name
        x_unit = self.units.get(x_name, '')
        x_selected_unit = options.get('x unit', 'auto')
        x_scale, _x_unit = self.scale_axis(x, x_name, x_unit, x_selected_unit)

        y = self.y(options)
        y_name = y.name
        y_unit = self.units.get(y_name, '')
        y_selected_unit = options.get('y unit', 'auto')
        y_scale, _y_unit = self.scale_axis(y, y_name, y_unit, y_selected_unit)
        return x_scale, y_scale

    def xlim(self, xrange=None, segment=0):
        upper_limit = max(self.x())-2*self.delta_x
        lower_limit = min(self.x())+2*self.delta_x
        if xrange == None:
            xrange = [lower_limit,upper_limit]
        if xrange[0] < lower_limit:
            xrange[0] = lower_limit
        if xrange[1] > upper_limit:
            xrange[1] = upper_limit
        x = self.x().values
        ind1 = x >= xrange[0]
        ind2 = x <= xrange[1]
        ind = ind1 & ind2
        sections = []
        if segment != 0:
            current_segment = 0
            for i in range(len(ind)-1):
                if not ind[i] and ind[i+1]:
                    current_segment += 1
                if ind[i] and current_segment != segment:
                        ind[i] = False
        x = x[ind]
        y = self.y().values[ind]
        return x, y

    def invert_current(self):
        """
        Multiply all current-like columns by -1 in place.

        Returns
        -------
        list[str]
            Names of columns that were inverted.
        """
        inverted_columns = []

        for col in self.data.columns:
            col_str = str(col).lower()
            if col_str.startswith("current"):
                self.data[col] *= -1
                inverted_columns.append(col)

        return inverted_columns

    @staticmethod
    def scale_axis(values, column_name, current_unit, selected_unit='auto'):
        """
        Given a base-unit y-array, determine the best scaling factor and unit prefix.
        Returns (scale_factor, unit_label)
        """
        if current_unit is not None and selected_unit is not None and not (
                selected_unit == 'auto' and 'potential' in column_name.lower()):
            if column_name.lower() in ('time', 't', 'duration'):
                scale, unit = scale_time_axis(values, current_unit, selected_unit)
            else:
                scale, unit = scale_axis(values, current_unit, selected_unit)
        else:
            unit = current_unit
            scale = 1

        return scale, unit

    @staticmethod
    def format_axis_label(axis_name, axis_unit, symbol_labels="auto"):
        axis_key = str(axis_name).strip().lower()
        if axis_key == "dimensionless potential":
            return r"$\theta = nF(E - E^0)/(RT)$"
        if axis_key == "dimensionless current":
            return r"$\Phi = I/(nFSC^*\sqrt{D nFv/(RT)})$"
        if str(axis_name).strip().lower() == "i/ip0":
            return "$i / i_p^0$"
        if _symbol_labels_enabled(symbol_labels):
            axis_name = _symbolized_axis_name(axis_name)
        if axis_unit:
            # Replace leading 'u' with Greek 'μ' for micro
            if axis_unit.startswith("u"):
                axis_unit = "μ" + axis_unit[1:]

            if " vs " in axis_name:
                base, ref = axis_name.split(" vs ", 1)
                if "+" in ref:
                    ref = _format_reference_label_mathtext(ref)
                return f"{base} ({axis_unit} vs {ref})"
            else:
                return f"{axis_name} ({axis_unit})"
        else:
            return axis_name

    @staticmethod
    def format_derivative_axis_label(y_name, y_unit, x_name, x_unit, derivative):
        derivative = int(derivative)
        y_label = str(y_name)
        x_label = str(x_name)
        y_key = y_label.strip().lower()
        x_key = x_label.strip().lower()

        if y_key.startswith("current") and x_key.startswith("potential"):
            y_label = "i"
            x_label = "E"
            compact_echem = True
        else:
            compact_echem = False

        if derivative == 1:
            label = f"d{y_label}/d{x_label}" if compact_echem else f"d({y_label})/d({x_label})"
            if y_unit and x_unit:
                return f"{label} ({y_unit}/{x_unit})"
            if x_unit:
                return f"{label} (1/{x_unit})"
            return label

        if compact_echem:
            label = f"d$^{derivative}${y_label}/d{x_label}$^{derivative}$"
        else:
            label = f"d$^{derivative}$({y_label})/d({x_label})$^{derivative}$"
        if y_unit and x_unit:
            return f"{label} ({y_unit}/{x_unit}$^{derivative}$)"
        if x_unit:
            return f"{label} (1/{x_unit}$^{derivative}$)"
        return label

    def _plot_helper(self, x, y, options):
        """
        Shared helper for echem.plot and cv.plot.
        """
        # new figure
        if options.get('new plot'):
            fig, ax1 = plt.subplots()
        else:
            ax1 = plt.gca()
            fig = ax1.get_figure()

        # integrate
        if options.get('integrate'):
            y = np.cumsum(y)

        # Scale and label x axis
        x_name = self.x(options).name
        x_unit = self.units.get(x_name, '')
        selected_unit = options.get('x unit', 'auto')  # None, 'auto', or explicit
        x_scale, x_unit = self.scale_axis(x, x_name, x_unit, selected_unit)
        x = np.asarray(x, dtype=float) * x_scale
        if not options.get('xlabel'):
            norm_axes = getattr(self, "normalization_axes", {}) or {}
            norm_labels = getattr(self, "normalization_axis_labels", {}) or {}
            if x_name == norm_axes.get("x"):
                options['xlabel'] = norm_labels.get("x", self.format_axis_label(x_name, x_unit, options.get("symbol labels", "auto")))
            else:
                options['xlabel'] = self.format_axis_label(x_name, x_unit, options.get("symbol labels", "auto"))

        # Scale and label y axis
        y_name = self.y(options).name
        y_unit = self.units.get(y_name, '')
        selected_unit = options.get('y unit', 'auto')  # None, 'auto', or explicit
        y_scale, y_unit = self.scale_axis(y, y_name, y_unit, selected_unit)
        y = y * y_scale + options.get('offset',0)
        if not options.get('ylabel'):
            norm_axes = getattr(self, "normalization_axes", {}) or {}
            norm_labels = getattr(self, "normalization_axis_labels", {}) or {}
            derivative = options.get("derivative", 0)
            derivative = 0 if derivative is None else derivative
            if derivative != 0:
                options['ylabel'] = self.format_derivative_axis_label(
                    y_name,
                    y_unit,
                    x_name,
                    x_unit,
                    derivative,
                )
            elif y_name == norm_axes.get("y"):
                options['ylabel'] = norm_labels.get("y", self.format_axis_label(y_name, y_unit, options.get("symbol labels", "auto")))
            else:
                options['ylabel'] = self.format_axis_label(y_name, y_unit, options.get("symbol labels", "auto"))

        # ensure label
        label = options.get('label') or self.name
        label = apply_text_alterations(label, options.get("label alterations"))
        label = format_chemical_formulas(label)
        options['label'] = label

        # core plotting call
        plot_kwargs = options.get("plot options", {})
        plot_kwargs['color'] = options.get('color', 'k')
        plot_kwargs['label'] = options.get('label')
        plot_obj = plt.plot(x, y, **plot_kwargs)

        ax = plt.gca()

        # flip y-axis only if requested and currently not inverted
        if options.get('y flip'):
            y0, y1 = ax.get_ylim()
            # normal is y0 < y1; inverted if y0 > y1
            if y0 < y1:
                ax.invert_yaxis()

        # flip x-axis for US CV convention only if currently not inverted
        if self.type == 'Cyclic Voltammetry' and options.get('plot convention') == 'US':
            x0, x1 = ax.get_xlim()
            if x0 < x1:
                ax.invert_xaxis()

        # auto margins
        plt.margins(x=0.05, y=0.05)

        # set axis labels
        if options.get('xlabel'):
            plt.xlabel(options['xlabel'])
        if options.get('ylabel'):
            plt.ylabel(options['ylabel'])

        # title
        title_opt = options.get('title', True)
        title, subtitle = _resolve_single_plot_title_subtitle(self, options)
        title_fs = options.get("title fontsize")
        if title_fs in (None, "auto"):
            title_fs = _resolve_title_fontsize(title)
        subtitle_fs = options.get("subtitle fontsize")
        if subtitle_fs in (None, "auto"):
            subtitle_fs = _resolve_subtitle_fontsize(subtitle)
        if title_opt:
            _apply_plot_titles(fig, ax1, title, subtitle, title_fs, subtitle_fs)

        # legend
        if _plot_legend_requested(options, ax):
            plt.legend(fontsize=options.get('legend fontsize') or _default_legend_fontsize())

        _apply_ecat_axis_style(ax, options)
        _add_scale_bar(ax, options, unit=y_unit)

        # animate if requested
        if options.get('animate'):
            return animate(plot_obj[0],
                           rate=options.get('scan_rate') or getattr(self, 'scan_rate', None),
                           minrate=options.get('animate minrate'),
                           repeat=options.get('animate repeat'))

        return ax1

    def plot(self, options={}, **mpl_kwargs):
        """Plot one electrochemistry object.
        
        Parameters
        ----------
        options : dict or PlotOptions, optional
            Axis, scaling, derivative, segment, legend, and title options. See ``e.describe_options("plot")``.
        **mpl_kwargs
            Additional keyword arguments passed to Matplotlib.
        
        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the plotted trace.
        
        Examples
        --------
        >>> cv_obj.plot({"segment": 1})
        """
        options = PlotOptions.from_options(options).to_legacy_dict()
        options.update(mpl_kwargs)

        # data
        options['one column'] = True
        x = self.x(options).values
        y = self.y(options).values

        # plot
        return self._plot_helper(x, y, options)

    def get_point(self,guess_x):
        x = self.x(options).values
        y = self.y(options).values
        index = np.argmin(np.abs(x - guess_x))
        return x[index], y[index], index


    def stats(self,options={}):
        """Return basic metadata and numeric ranges for one electrochemistry object.
        
        Parameters
        ----------
        options : dict, optional
            Display options. See ``e.describe_options("plot")`` for common axis-display choices.
        
        Returns
        -------
        dict
            Statistics and metadata for the object.
        
        Examples
        --------
        >>> cv_obj.stats()
        """
        name = self.name
        options['one column'] = True
        x = self.x(options)
        start_x = x.values[0]
        end_x = x.values[-1]
        min_x = min(x.values)
        max_x = max(x.values)
        if self.segments is None:
            segments = round_sigfigs(len(x.values) / ((max_x - min_x) / self.delta_x),2)
        else:
            segments = self.segments
        return {
            'solvent': self.solvent,
            'gas': self.gas,
            'compounds': self.compounds,
            'concentrations': self.concentrations,
            'start_x':start_x,
            'end_x':end_x,
            'min_x':min_x,
            'max_x':max_x,
            'delta_x':self.delta_x,
            'segments':segments
            }

    def info(self):
        """Print a compact information summary for one electrochemistry object.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            Prints information to the notebook output.
        
        Examples
        --------
        >>> cv_obj.info()
        """
        info = {
            'name': getattr(self, 'name', None),
            'folder path': getattr(self, 'folderpath', getattr(self, 'folder_path', None)),
            'timestamp': getattr(self, 'timestamp', None),
            'creation time': getattr(self, 'creation_time', None),
            'modification time': getattr(self, 'modification_time', None),
            'reference shift': getattr(self, 'reference_shift', None),
            'reference label': getattr(self, 'reference_label', None),
            'reference mode': getattr(self, 'reference_mode', None),
            'reference source file': getattr(self, 'reference_source_file', None),
            'ir comp resistance': getattr(self, 'ir_comp_resistance', None),
            'ir uncomp resistance': getattr(self, 'ir_uncomp_resistance', None),
            'ir comp percent': getattr(self, 'ir_comp_percent', None),
        }
        stats = self.stats()
        info.update(stats)
        return info

    def combine_concs_chems(self, concentrations, compounds, options={'separate concentration': True}):
        if not compounds:
            return ''

        conc_and_chem = []
        for conc, comp in zip(concentrations + [''] * len(compounds), compounds):
            conc_and_chem.append(f"{conc + ' ' if conc else ''}{comp}")

        return conc_and_chem

    def txt_stats(self, options=None):
        if options is None:
            options = {}

        stats = self.stats().copy()

        stats["exp type"] = _exp_type_short(getattr(self, "type", ""))

        stats["compounds"] = self.combine_concs_chems(
            stats.get("concentrations", []),
            stats.get("compounds", []),
            options,
        )

        stats.pop("concentrations", None)
        return stats

    def reformat_label(self,label):
        if '/' in label:
            index = label.index('/')
            label = label[:index] + ' (' + label[index+1:] + ')'
        return label

class cv(echem):
    """Cyclic voltammetry object with CV-specific plotting and analysis methods.
    
    Parameters
    ----------
    filepath : str or path-like, optional
        CV text file to parse.
    options : dict or ImportOptions, optional
        Import and parser options. See ``e.describe_options("get_data")``.
    
    Examples
    --------
    >>> cv_obj = e.cv(path, {"software": "CH"})
    """
    def __init__(self, filepath=None, options={}):
        super().__init__(filepath, options)
        self.type = "Cyclic Voltammetry"
        self.get_data_from_file(filepath, options)  # parse scan rate, potentials, etc.

    def remove_parentheses_and_replace_last_space(self, input_string):
            # Remove all open and closing parentheses
            cleaned_string = input_string.replace("(", "").replace(")", "")

            # Find the index of the last space in the cleaned string
            last_space_index = cleaned_string.rfind(" ")

            if last_space_index != -1:
                # Replace the last space with a "/"
                cleaned_string = cleaned_string[:last_space_index] + "/" + cleaned_string[last_space_index + 1:]

            return cleaned_string
        
    # manual init
    def manual_init(self, name, data, options={}):
        # set values
        self.name = str(name)
        self.data = data
        self.num_x_cols = 1

        # reformat data headers
        new_column_names = {
            data.columns[0][1]: self.remove_parentheses_and_replace_last_space(data.columns[0][1]),
            data.columns[1][1]: self.remove_parentheses_and_replace_last_space(data.columns[1][1])
        }
        self.data = self.data.rename(columns=new_column_names,level=1)

        # Manual constructors often start from a two-level column index; flatten
        # to the same public column names used by file-backed objects first.
        self.data.columns = [col[-1] if isinstance(col, tuple) else col for col in self.data.columns]
        self.data = self.data.rename(
            columns={
                col: col.split("/", 1)[0]
                for col in self.data.columns
                if isinstance(col, str) and "/" in col
            }
        )
        self.units["Potential"] = "V"
        self.units["Current"] = "A"

        self.get_data_from_name()
        self.type = "Cyclic Voltammetry"

        self.init_E = self.x().iloc[0]
        self.final_E = self.x().iloc[-1]
        self.min_E = np.min(self.x())
        self.max_E = np.max(self.x())
        self.segments = count_segments(self.x())
        self.delta_x = np.abs(self.x().iloc[1] - self.x().iloc[0])
        self.temperature = 298
        self.electrode_area = 0

        # get scan rate
        try:
            self.scan_rate = float(name)
        except ValueError:
            match = re.search(r'(\d+\.?\d*)[numμ]?Vs', name)[0]
            if match:
                # Extract the numeric part and convert it to a float (e.g., "100" to 0.1, "200" to 0.2)
                value = float(re.search(r'(\d+\.?\d*)', match)[0])
                prefix = re.search(r'[numμ]?Vs', match)[0][0]
                conversion_dict = {
                    "n": 1e-9,  # Nano (10^-9)
                    "u": 1e-6,  # Micro (10^-6)
                    "m": 1e-3,   # Milli (10^-3)
                }
                self.scan_rate = value * conversion_dict.get(prefix, 1)

        self.modify_by_options(options)

    def get_data_from_file(self, filepath, options):
        software_parsers = {
            "CH": self._parse_ch_file,
            "BASI": self._parse_basi_file,
            "EC-Lab": self._parse_eclab_file,
        }

        parser = software_parsers.get(self.software)
        if parser is not None:
            parser(filepath, options)
        else:
            raise ValueError(f"No parser implemented for software type: {self.software}")

    def _parse_ch_file(self, filepath, options):
        """
        Parses CH file metadata directly from lines without using pandas.
        """
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()

        def get_value(key):
            for line in lines:
                if key in line:
                    try:
                        return float(line.split('=')[1].strip())
                    except Exception:
                        return None
            return None

        self._parse_ir_compensation_from_lines(lines)
        self.init_E = get_value('Init E')
        self.max_E = get_value('High E')
        self.min_E = get_value('Low E')
        self.scan_rate = get_value('Scan Rate')
        self.segments = int(get_value('Segment')) if get_value('Segment') is not None else None
        self.quiet_time = _parse_quiet_time_from_lines(lines)
        self.sample_int = get_value('Sample Interval')
        self.sensitivity = get_value('Sensitivity')

    def _parse_basi_file(self, filepath, options):
        """
        Parses BASI file metadata directly from lines without using pandas.
        """
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            lines = [line.strip() for line in f.readlines()]

        initial_potential = switching_potential1 = switching_potential2 = final_potential = None

        end_params = next((i for i, line in enumerate(lines) if line == '[Begin Data]'), len(lines))
        self.quiet_time = _parse_quiet_time_from_lines(lines[:end_params])

        for line in lines[:end_params]:
            if m := re.search(r'Initial Potential\s*:\s*([\d.-]+) mV', line):
                initial_potential = float(m.group(1)) / 1000
            elif m := re.search(r'Switching Potential 1\s*:\s*([\d.-]+) mV', line):
                switching_potential1 = float(m.group(1)) / 1000
            elif m := re.search(r'Switching Potential 2\s*:\s*([\d.-]+) mV', line):
                switching_potential2 = float(m.group(1)) / 1000
            elif m := re.search(r'Final Potential\s*:\s*([\d.-]+) mV', line):
                final_potential = float(m.group(1)) / 1000
            elif m := re.search(r'Scan Rate\s*:\s*([\d.-]+) mV/s', line):
                self.scan_rate = float(m.group(1)) / 1000
            elif m := re.search(r'Number of segments\s*:\s*([\d.-]+)', line):
                self.segments = int(m.group(1))
            elif m := re.search(r'Sample Interval\s*:\s*([\d.-]+) mV', line):
                self.sample_int = float(m.group(1)) / 1000
            elif m := re.search(r'IR-Comp. Value\s*:\s*([\d.-]+) Ohm', line):
                self.IR_comp = float(m.group(1))

        potential_values = [v for v in [initial_potential, switching_potential1, switching_potential2, final_potential]
                            if v is not None]
        self.init_E = initial_potential
        self.min_E = min(potential_values) if potential_values else None
        self.max_E = max(potential_values) if potential_values else None
        self.final_E = final_potential
        self.sensitivity = "N/A"

    def _parse_eclab_file(self, filepath, options):
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()

        if len(lines) >= 4:
            self.type = lines[3].strip()
        self.quiet_time = _parse_quiet_time_from_lines(lines[:60])

        # Track if the previous line was the dE/dt value
        prev_line_was_scanrate = False
        scanrate_value = None

        for i, line in enumerate(lines[:60]):
            line = line.strip()

            if line.startswith("Ei (V)"):
                self.init_E = float(line.rpartition(" ")[-1].strip())
            elif line.startswith("Ef (V)"):
                self.final_E = float(line.rpartition(" ")[-1].strip())
            elif line.startswith("E1 (V)"):
                self.max_E = float(line.rpartition(" ")[-1].strip())
            elif line.startswith("E2 (V)"):
                self.min_E = float(line.rpartition(" ")[-1].strip())
            elif line.startswith("N") and "Step percent" not in line:
                try:
                    self.segments = int(line.rpartition(" ")[-1].strip())
                except Exception:
                    pass
            elif line.startswith("dE/dt") and "unit" not in line:
                try:
                    scanrate_value = float(line.rpartition(" ")[-1].strip())
                    prev_line_was_scanrate = True
                except Exception:
                    pass

            # Capture scan rate unit (on the next line)
            elif prev_line_was_scanrate:
                prev_line_was_scanrate = False
                unit = line.lower()
                if "mv/s" in unit:
                    self.scan_rate = scanrate_value / 1000  # convert to V/s
                elif "v/s" in unit:
                    self.scan_rate = scanrate_value
                else:
                    self.scan_rate = scanrate_value  # fallback


    def x(self, options={}):
        """Return selected CV x-axis data, usually potential.
        
        Parameters
        ----------
        options : dict or PlotOptions, optional
            Axis, segment, derivative, and scaling options. See ``e.describe_options("plot")``.
        
        Returns
        -------
        pandas.Series
            Selected CV x-axis data.
        
        Examples
        --------
        >>> potential = cv_obj.x({"segment": 1})
        """
        options = {} if options is None else dict(options)
        if options.get('normalize', False):
            raise ValueError(
                "Plot-time normalize=True has been removed. "
                "Use normalize(...) first, then plot the returned CV(s)."
            )
        requested_x_axis = options.get('x axis', None)
        x_col_name = requested_x_axis
        if x_col_name is None:
            normalized_axis = _default_normalized_axis(self, "x")
            if normalized_axis is not None:
                x_col_name = normalized_axis
            elif self.has_reference_shift():
                x_col_name = self.reference_axis_name()
            elif "potential" in [col.lower() for col in self.data.columns]:
                x_col_name = "Potential"
            else:
                raise ValueError(
                    "No suitable x column found. "
                    f"Available dataframe columns: {list(self.data.columns)}"
                )
        if requested_x_axis is None and not options.get("one column", True):
            options.pop('x axis', None)
        else:
            options['x axis'] = x_col_name

        x = super().x(options)
        return x

    def y(self, options={}):
        """Return selected CV y-axis data, usually current.
        
        Parameters
        ----------
        options : dict or PlotOptions, optional
            Axis, segment, derivative, current-density, and normalization options. See ``e.describe_options("plot")``.
        
        Returns
        -------
        pandas.Series
            Selected CV y-axis data.
        
        Examples
        --------
        >>> current = cv_obj.y({"y axis": "i/ip0", "ip0": 1e-5})
        """
        options = {} if options is None else dict(options)
        if options.get('normalize', False):
            raise ValueError(
                "Plot-time normalize=True has been removed. "
                "Use normalize(...) first, then plot the returned CV(s)."
            )
        options['y axis'] = options.get(
            'y axis',
            _default_normalized_axis(self, "y") or 'Current',
        )

        requested_y_axis = str(options.get('y axis', '')).strip().lower()
        if requested_y_axis.startswith('current density'):
            existing_density_col = None
            for col in self.data.columns:
                if str(col).strip().lower().startswith('current density'):
                    existing_density_col = col
                    break

            if existing_density_col is not None:
                density_options = options.copy()
                density_options['y axis'] = existing_density_col
                y = super().y(density_options)
            else:
                if getattr(self, 'electrode_area', 0) in (None, 0):
                    raise ValueError(
                        "Cannot plot current density without a nonzero electrode area."
                    )
                current_options = options.copy()
                current_options['y axis'] = 'Current'
                y = super().y(current_options) / self.electrode_area
                y.name = 'Current Density'
                current_unit = self.units.get('Current', 'A')
                self.units.setdefault('Current Density', f"{current_unit}/cm$^2$")
        elif _is_ip0_y_axis(requested_y_axis):
            existing_ip0_col = _find_column_by_text(self.data.columns, "i/ip0")

            if existing_ip0_col is not None:
                ip0_options = options.copy()
                ip0_options['y axis'] = existing_ip0_col
                y = super().y(ip0_options)
            else:
                ip0 = _resolve_ip0_values([self], options)[0]
                current_options = options.copy()
                current_options['y axis'] = 'Current'
                y = super().y(current_options) / ip0
                y.name = 'i/ip0'
                self.units.setdefault('i/ip0', '')
        else:
            y = super().y(options)

        return y

    def xy(self, options={}):
        """Return selected CV x and y data series.
        
        Parameters
        ----------
        options : dict or PlotOptions, optional
            Axis, segment, derivative, and normalization options. See ``e.describe_options("plot")``.
        
        Returns
        -------
        tuple of pandas.Series
            Selected x and y data.
        
        Examples
        --------
        >>> x, y = cv_obj.xy({"segment": 1})
        """
        x = self.x(options).values
        y = self.y(options).values
        return x,y

    def trim(self, potential_window=None, options=None, **kwargs):
        """Return a CV trimmed to a potential window.

        By default, disconnected pointwise windows are expanded to preserve a
        connected CV waveform. Use ``{"mode": "pointwise"}`` for pointwise
        trimming or ``{"mode": "strict"}`` to reject
        disconnected windows.
        """
        trim_options = {}
        if isinstance(potential_window, TrimOptions) and options is None and not kwargs:
            trim_options = potential_window.to_legacy_dict()
        elif isinstance(potential_window, dict) and options is None:
            trim_options.update(potential_window)
        else:
            if potential_window is not None:
                trim_options["potential window"] = potential_window
            if options is not None:
                trim_options.update(options)
        trim_options.update(kwargs)
        typed_options = TrimOptions.from_options(trim_options)
        trim_options = typed_options.to_legacy_dict()

        window = trim_options.get("potential window")

        x_values = self.x(trim_options).to_numpy(dtype=float)
        window_info = _cv_trim_window_info(x_values, window, trim_options)
        mask = window_info["mask"]
        if not np.any(mask):
            raise ValueError("cv.trim selected no points.")

        target = self if trim_options.get("inplace", False) else deepcopy(self)
        target.data = self.data.loc[mask].reset_index(drop=True).copy()
        target.units = getattr(self, "units", {}).copy()
        target.data.attrs["units"] = target.units
        target.trim_metadata = {
            "potential_window": list(window),
            "potential_window_requested": window_info["requested"],
            "potential_window_effective": window_info["effective"],
            "mode": window_info["mode"],
            "window_expanded": window_info["expanded"],
            "window_break_count": window_info["break_count"],
        }
        target._refresh_cv_metadata_from_data(trim_options)
        return target

    def _refresh_cv_metadata_from_data(self, options=None):
        options = {} if options is None else dict(options)
        if self.data.empty:
            return

        x = self.x(options)
        if len(x) == 0:
            return

        self.init_E = float(x.iloc[0])
        self.final_E = float(x.iloc[-1])
        self.min_E = float(np.nanmin(x))
        self.max_E = float(np.nanmax(x))
        self.segments = count_segments(x)
        self.delta_x = float(abs(x.iloc[1] - x.iloc[0])) if len(x) > 1 else np.nan

    def plot(self, options={}):
        """Plot a cyclic voltammogram.
        
        Parameters
        ----------
        options : dict or PlotOptions, optional
            Axis, segment, derivative, normalization, legend, and title options. See ``e.describe_options("plot")``.
        
        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the CV plot.
        
        Examples
        --------
        >>> cv_obj.plot({"y axis": "i/ip0", "ip0": 1e-5})
        """
        raw_options = {} if options is None else options
        has_explicit_new_plot = False
        if isinstance(raw_options, dict):
            has_explicit_new_plot = any(
                str(key).strip().lower().replace("_", " ") == "new plot"
                for key in raw_options
            )
        elif hasattr(raw_options, "new_plot"):
            has_explicit_new_plot = True

        options = PlotOptions.from_options(options).to_legacy_dict()
        if not has_explicit_new_plot:
            options["new plot"] = True

        # data
        options['one column'] = True
        x = self.x(options).values
        y = self.y(options).values

        segment_color_mode = self._normalize_segment_color_mode(
            options.get("segment color mode", "off")
        )
        if segment_color_mode != "off":
            return self._plot_colored_segments(x, y, options, segment_color_mode)

        segments = options.get('plot segments') or options.get('plot segment')
        if segments is not None:
            x, y = self._select_segments(x, y, segments)

        # plot
        return self._plot_helper(x, y, options)

    @staticmethod
    def _normalize_segment_color_mode(mode):
        text = str(mode).strip().lower().replace("_", " ").replace("-", " ")
        if text in {"", "none", "false", "0"}:
            return "off"
        if text in {"auto", "off", "discrete", "discrete gradient", "continuous gradient"}:
            return text
        raise ValueError(
            "'segment color mode' must be 'auto', 'off', 'discrete', "
            "'discrete gradient', or 'continuous gradient'."
        )

    @staticmethod
    def _split_segment_arrays(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        dx = np.diff(x)
        change = np.where(np.diff(np.sign(dx)))[0] + 1
        split_x = np.split(x, change)
        split_y = np.split(y, change)
        return [
            {"number": idx + 1, "x": sx, "y": sy}
            for idx, (sx, sy) in enumerate(zip(split_x, split_y))
        ]

    @staticmethod
    def _selected_segment_numbers(selection, available_numbers):
        if selection is None:
            return list(available_numbers)
        if isinstance(selection, int):
            requested = [selection]
        elif isinstance(selection, (list, tuple, np.ndarray)):
            requested = [int(item) for item in selection]
        else:
            raise TypeError("segments must be int, list[int], or None")
        available = set(available_numbers)
        return [number for number in requested if number in available]

    @staticmethod
    def _format_segment_group_label(group):
        group = [int(item) for item in group]
        if len(group) == 1:
            return f"Segment {group[0]}"
        sorted_group = sorted(group)
        contiguous = sorted_group == list(range(sorted_group[0], sorted_group[-1] + 1))
        if contiguous:
            return f"Segment {sorted_group[0]}-{sorted_group[-1]}"
        return "Segments " + ", ".join(str(item) for item in group)

    @staticmethod
    def _resolve_segment_color_groups(visible_numbers, group_spec):
        visible_numbers = [int(number) for number in visible_numbers]
        if not visible_numbers:
            return []

        if group_spec is None:
            group_spec = 2

        if isinstance(group_spec, int):
            group_size = max(1, int(group_spec))
            groups = [
                visible_numbers[idx:idx + group_size]
                for idx in range(0, len(visible_numbers), group_size)
            ]
            if len(groups) > 1 and len(groups[-1]) < group_size:
                groups[-2].extend(groups[-1])
                groups.pop()
            return groups

        if isinstance(group_spec, (list, tuple, np.ndarray)):
            visible = set(visible_numbers)
            groups = []
            for item in group_spec:
                if isinstance(item, int):
                    group = [item]
                elif isinstance(item, (list, tuple, np.ndarray)):
                    group = [int(segment) for segment in item]
                else:
                    raise TypeError("'segment color groups' entries must be ints or lists of ints.")
                group = [segment for segment in group if segment in visible]
                if group:
                    groups.append(group)
            grouped = {segment for group in groups for segment in group}
            for segment in visible_numbers:
                if segment not in grouped:
                    groups.append([segment])
            return groups

        raise TypeError("'segment color groups' must be an int or a list of segment groups.")

    @staticmethod
    def _segment_colormap(options):
        colors = options.get("gradient colors") or []
        if colors:
            cmap = mpl.colors.LinearSegmentedColormap.from_list("ecat_segment_gradient", colors)
        else:
            cmap = plt.get_cmap(
                options.get("gradient colormap")
                or options.get("default gradient colormap", "viridis")
            )
        if options.get("gradient reverse", False):
            cmap = cmap.reversed()
        return cmap

    @staticmethod
    def _segment_discrete_colors(n_colors, options):
        palette = options.get("gradient colors") or options.get("default colors") or ["black"]
        return [palette[idx % len(palette)] for idx in range(n_colors)]

    @staticmethod
    def _segment_gradient_colors(n_colors, options):
        if n_colors <= 1:
            values = [0.5]
        else:
            values = np.linspace(0, 1, n_colors)
        cmap = cv._segment_colormap(options)
        return [cmap(value) for value in values]

    @staticmethod
    def _segment_colorbar_ticks(labels, options):
        tick_mode = str(options.get("colorbar tick labels", "endpoints")).strip().lower()
        ticks = np.arange(1, len(labels) + 1, dtype=float)
        if tick_mode == "all" or len(labels) <= 2:
            return ticks, labels
        tick_labels = [""] * len(labels)
        tick_labels[0] = labels[0]
        tick_labels[-1] = labels[-1]
        return ticks, tick_labels

    @staticmethod
    def _segment_colorbar_tick_label(label):
        text = str(label)
        if text.startswith("Segments "):
            return text.removeprefix("Segments ").strip()
        if text.startswith("Segment "):
            return text.removeprefix("Segment ").strip()
        return text

    @staticmethod
    def _segment_colorbar_spec(colors, labels, options, segment_color_mode=None):
        if not labels:
            return {"plot labels": [], "gradient groups": [], "discrete indices": []}
        colorbar_style = str(options.get("colorbar style", "auto")).strip().lower()
        colorbar_style = colorbar_style.replace("_", " ").replace("-", " ")
        if colorbar_style == "auto":
            mode = str(segment_color_mode or "").strip().lower().replace("_", " ").replace("-", " ")
            colorbar_style = "continuous" if mode == "continuous gradient" else "discrete"
        numeric_labels = [cv._segment_colorbar_tick_label(label) for label in labels]
        if colorbar_style in {"discrete", "swatch", "swatches"}:
            cmap = mpl.colors.ListedColormap(colors or ["black"])
            boundaries = np.arange(0.5, len(labels) + 1.5, dtype=float)
            norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
        elif len(colors) <= 1:
            cmap = mpl.colors.ListedColormap(colors or ["black"])
            norm = mpl.colors.Normalize(vmin=1, vmax=1 + 1e-12)
        else:
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                "ecat_segment_colorbar",
                colors,
                N=256,
            )
            norm = mpl.colors.Normalize(vmin=1, vmax=len(labels))
        ticks, tick_labels = cv._segment_colorbar_ticks(numeric_labels, options)
        endpoint_ticks = [ticks[0], ticks[-1]] if len(ticks) > 1 else list(ticks)
        endpoint_ticklabels = [numeric_labels[0], numeric_labels[-1]] if len(numeric_labels) > 1 else list(numeric_labels)
        return {
            "plot labels": ["_nolegend_"] * len(labels),
            "gradient groups": [{
                "indices": list(range(len(labels))),
                "values": np.arange(1, len(labels) + 1, dtype=float),
                "cmap": cmap,
                "norm": norm,
                "ticks": ticks,
                "ticklabels": tick_labels,
                "endpoint ticks": endpoint_ticks,
                "endpoint ticklabels": endpoint_ticklabels,
                "legend title": "Segment",
                "legend unit": "",
                "legend context line": "Segments",
            }],
            "discrete indices": [],
        }

    @staticmethod
    def _draw_segment_colorbar_legend(ax, colors, labels, options, segment_color_mode=None):
        if len(labels) <= 1:
            return None
        color_spec = cv._segment_colorbar_spec(colors, labels, options, segment_color_mode=segment_color_mode)
        legend_fs = options.get("legend fontsize", None)
        if legend_fs in (None, "auto"):
            legend_fs, legend_loc, legend_outside = _resolve_adaptive_legend_layout(
                ax,
                color_spec,
                options,
            )
        else:
            legend_loc = _normalize_legend_loc(options.get("legend loc", "best"))
            legend_outside = options.get("legend outside", False)
        legend_options = options.copy()
        legend_options["legend fontsize"] = legend_fs
        legend_options["legend loc"] = legend_loc
        legend_options["legend outside"] = legend_outside
        return _draw_multiplot_legend_and_colorbars(ax, color_spec, legend_options, legend_fs)

    def _scale_segment_plot_data(self, segments, options):
        x_name = self.x(options).name
        x_unit = self.units.get(x_name, '')
        all_x = np.concatenate([segment["x"] for segment in segments]) if segments else np.asarray([])
        x_scale, x_unit = self.scale_axis(all_x, x_name, x_unit, options.get('x unit', 'auto'))

        y_name = self.y(options).name
        y_unit = self.units.get(y_name, '')
        all_y = np.concatenate([segment["y"] for segment in segments]) if segments else np.asarray([])
        y_scale, y_unit = self.scale_axis(all_y, y_name, y_unit, options.get('y unit', 'auto'))

        if not options.get('xlabel'):
            norm_axes = getattr(self, "normalization_axes", {}) or {}
            norm_labels = getattr(self, "normalization_axis_labels", {}) or {}
            if x_name == norm_axes.get("x"):
                options['xlabel'] = norm_labels.get("x", self.format_axis_label(x_name, x_unit, options.get("symbol labels", "auto")))
            else:
                options['xlabel'] = self.format_axis_label(x_name, x_unit, options.get("symbol labels", "auto"))

        if not options.get('ylabel'):
            norm_axes = getattr(self, "normalization_axes", {}) or {}
            norm_labels = getattr(self, "normalization_axis_labels", {}) or {}
            derivative = options.get("derivative", 0)
            derivative = 0 if derivative is None else derivative
            if derivative != 0:
                options['ylabel'] = self.format_derivative_axis_label(
                    y_name,
                    y_unit,
                    x_name,
                    x_unit,
                    derivative,
                )
            elif y_name == norm_axes.get("y"):
                options['ylabel'] = norm_labels.get("y", self.format_axis_label(y_name, y_unit, options.get("symbol labels", "auto")))
            else:
                options['ylabel'] = self.format_axis_label(y_name, y_unit, options.get("symbol labels", "auto"))

        scaled = []
        for segment in segments:
            scaled.append({
                "number": segment["number"],
                "x": segment["x"] * x_scale,
                "y": segment["y"] * y_scale + options.get("offset", 0),
            })
        return scaled

    def _finish_segment_colored_plot(self, fig, ax, options):
        if options.get('y flip'):
            y0, y1 = ax.get_ylim()
            if y0 < y1:
                ax.invert_yaxis()

        if self.type == 'Cyclic Voltammetry' and options.get('plot convention') == 'US':
            x0, x1 = ax.get_xlim()
            if x0 < x1:
                ax.invert_xaxis()

        plt.margins(x=0.05, y=0.05)

        if options.get('xlabel'):
            ax.set_xlabel(options['xlabel'])
        if options.get('ylabel'):
            ax.set_ylabel(options['ylabel'])

        title_opt = options.get('title', True)
        title, subtitle = _resolve_single_plot_title_subtitle(self, options)
        title_fs = options.get("title fontsize")
        if title_fs in (None, "auto"):
            title_fs = _resolve_title_fontsize(title)
        subtitle_fs = options.get("subtitle fontsize")
        if subtitle_fs in (None, "auto"):
            subtitle_fs = _resolve_subtitle_fontsize(subtitle)
        if title_opt:
            _apply_plot_titles(fig, ax, title, subtitle, title_fs, subtitle_fs)

        return fig, ax

    def _plot_colored_segments(self, x, y, options, segment_color_mode):
        if options.get("animate"):
            raise ValueError("Segment-colored cv.plot does not support animation.")

        all_segments = self._split_segment_arrays(x, y)
        available_numbers = [segment["number"] for segment in all_segments]
        selection = options.get('plot segments') or options.get('plot segment')
        selected_numbers = self._selected_segment_numbers(selection, available_numbers)
        selected = [segment for segment in all_segments if segment["number"] in selected_numbers]

        if segment_color_mode == "auto":
            segment_color_mode = "discrete gradient" if len(selected) > 3 else "off"
            if segment_color_mode == "off":
                if selection is not None:
                    x, y = self._select_segments(x, y, selection)
                return self._plot_helper(x, y, options)

        if options.get('new plot'):
            fig, ax = plt.subplots()
        else:
            ax = plt.gca()
            fig = ax.get_figure()

        if options.get("integrate"):
            running = 0
            integrated = []
            for segment in selected:
                y_int = np.cumsum(segment["y"]) + running
                if len(y_int):
                    running = y_int[-1]
                integrated.append({"number": segment["number"], "x": segment["x"], "y": y_int})
            selected = integrated

        selected = self._scale_segment_plot_data(selected, options)

        if segment_color_mode == "continuous gradient":
            legend_info = self._plot_continuous_segment_gradient(ax, selected, options)
        else:
            legend_info = self._plot_grouped_colored_segments(ax, selected, options, segment_color_mode)

        fig, ax = self._finish_segment_colored_plot(fig, ax, options)
        self._draw_segment_legend_after_finish(ax, legend_info, options)
        _apply_ecat_axis_style(ax, options)
        return ax

    def _plot_grouped_colored_segments(self, ax, selected, options, segment_color_mode):
        segment_by_number = {segment["number"]: segment for segment in selected}
        visible_numbers = [segment["number"] for segment in selected]
        groups = self._resolve_segment_color_groups(
            visible_numbers,
            options.get("segment color groups", 2),
        )
        labels = [self._format_segment_group_label(group) for group in groups]

        if segment_color_mode == "discrete":
            colors = self._segment_discrete_colors(len(groups), options)
            use_colorbar = False
        else:
            colors = self._segment_gradient_colors(len(groups), options)
            legend_mode = str(options.get("legend mode", "auto")).strip().lower()
            use_colorbar = _plot_legend_option_enabled(options.get("legend")) and legend_mode in {"auto", "colorbar"}

        plot_kwargs = dict(options.get("plot options", {}) or {})
        plot_kwargs.pop("color", None)
        plot_kwargs.pop("label", None)

        for group, label, color in zip(groups, labels, colors):
            first = True
            for segment_number in group:
                segment = segment_by_number.get(segment_number)
                if segment is None:
                    continue
                plot_label = "_nolegend_" if use_colorbar else (label if first else "_nolegend_")
                ax.plot(
                    segment["x"],
                    segment["y"],
                    color=color,
                    label=plot_label,
                    **plot_kwargs,
                )
                first = False

        if _plot_legend_option_enabled(options.get("legend")) and len(labels) > 1:
            legend_mode = str(options.get("legend mode", "auto")).strip().lower()
            if use_colorbar:
                return {
                    "type": "colorbar",
                    "colors": colors,
                    "labels": labels,
                    "segment color mode": segment_color_mode,
                }
            elif legend_mode in {"auto", "discrete"}:
                return {"type": "discrete"}
        return None

    def _plot_continuous_segment_gradient(self, ax, selected, options):
        line_segments = []
        color_values = []
        labels = [self._format_segment_group_label([segment["number"]]) for segment in selected]

        for display_idx, segment in enumerate(selected, start=1):
            points = np.column_stack([segment["x"], segment["y"]])
            if len(points) < 2:
                continue
            pieces = np.stack([points[:-1], points[1:]], axis=1)
            line_segments.extend(pieces)
            if len(pieces) == 1:
                color_values.append(float(display_idx))
            else:
                color_values.extend(np.linspace(display_idx, display_idx + 0.999, len(pieces)))

        if not line_segments:
            return None

        plot_kwargs = dict(options.get("plot options", {}) or {})
        linewidth = plot_kwargs.pop("linewidth", plot_kwargs.pop("linewidths", None))
        alpha = plot_kwargs.pop("alpha", None)
        linestyle = plot_kwargs.pop("linestyle", plot_kwargs.pop("linestyles", None))
        plot_kwargs.pop("color", None)
        plot_kwargs.pop("label", None)

        cmap = self._segment_colormap(options)
        norm = mpl.colors.Normalize(vmin=1, vmax=max(1, len(selected) + 1))
        collection = mpl.collections.LineCollection(
            line_segments,
            cmap=cmap,
            norm=norm,
            linewidths=linewidth if linewidth is not None else mpl.rcParams["lines.linewidth"],
            alpha=alpha,
            linestyles=linestyle if linestyle is not None else "solid",
            **plot_kwargs,
        )
        collection.set_array(np.asarray(color_values, dtype=float))
        ax.add_collection(collection)
        ax.autoscale_view()

        if _plot_legend_option_enabled(options.get("legend")) and len(labels) > 1:
            colors = self._segment_gradient_colors(max(1, len(labels)), options)
            return {
                "type": "colorbar",
                "colors": colors,
                "labels": labels,
                "segment color mode": "continuous gradient",
            }
        return None

    def _draw_segment_legend_after_finish(self, ax, legend_info, options):
        if not legend_info:
            return None

        if legend_info.get("type") == "colorbar":
            return self._draw_segment_colorbar_legend(
                ax,
                legend_info["colors"],
                legend_info["labels"],
                options,
                segment_color_mode=legend_info.get("segment color mode"),
            )

        if legend_info.get("type") == "discrete":
            return ax.legend(
                fontsize=options.get("legend fontsize") or _default_legend_fontsize(),
                loc=_normalize_legend_loc(options.get("legend loc", "best")),
            )

        return None

    def _plot_from_analysis_options(self, options):
        plot_options = _plot_options_from_mapping(options)
        return self.plot(plot_options)

    def _select_segments(self, x, y, seg_spec):
        """
        Return (x_sel, y_sel) with only the requested segment(s).

        seg_spec : int | list[int] | None
            * None  →  return original arrays
            * int   →  that single segment
            * list  →  concatenated segments in numeric order
        """
        if seg_spec is None:
            return x, y

        # split once
        dx = np.diff(x)
        change = np.where(np.diff(np.sign(dx)))[0] + 1
        seg_x = np.split(x, change)
        seg_y = np.split(y, change)

        def fetch(s):
            if 1 <= s <= len(seg_x):
                return seg_x[s - 1], seg_y[s - 1]
            print(f"Segment {s} out of range (1-{len(seg_x)}) -> skipped.")
            return np.empty(0), np.empty(0)

        if isinstance(seg_spec, int):
            return fetch(seg_spec)

        if isinstance(seg_spec, list) and all(isinstance(s, int) for s in seg_spec):
            xs, ys = zip(*(fetch(s) for s in seg_spec))
            return np.concatenate(xs), np.concatenate(ys)

        raise TypeError("segments must be int, list[int], or None")

    def xy_scale(self, options={}):
        x = self.x(options)
        x_name = x.name
        x_unit = self.units.get(x_name, '')
        x_selected_unit = options.get('x unit', 'auto')
        x_scale, x_unit = self.scale_axis(x, x_name, x_unit, x_selected_unit)

        y = self.y(options)
        y_name = y.name
        y_unit = self.units.get(y_name, '')
        y_selected_unit = options.get('y unit', 'auto')
        y_scale, y_unit = self.scale_axis(y, y_name, y_unit, y_selected_unit)
        return x_scale, y_scale

    def normalize(self, options={}):
        """Add physical dimensionless CV axes to this object and return ``self``.
        
        Parameters
        ----------
        options : dict or NormalizeOptions, optional
            Dimensionless CV normalization options. See ``e.describe_options("cv.normalize")``.
        
        Returns
        -------
        cv
            This mutated CV object.
        
        Examples
        --------
        >>> cv_obj.normalize({"E0": 0.0, "D": 1e-5, "C": 10, "C unit": "mM"})
        """
        typed_options = NormalizeOptions.from_options(options)
        normalized = _normalize_single_cv(self, typed_options.to_legacy_dict())
        self.__dict__.update(normalized.__dict__)
        return self

    def normalize_current(self, ip0, options={}):
        """Add or update the ``i/ip0`` column on this CV and return ``self``.
        
        Parameters
        ----------
        ip0 : float
            Nonzero reference peak current.
        options : dict, optional
            Display metadata for the normalized axis. See ``e.describe_options("cv.normalize_current")``.
        
        Returns
        -------
        cv
            This mutated CV object.
        
        Examples
        --------
        >>> cv_obj.normalize_current(2e-6)
        """
        _apply_normalized_current_axis(self, ip0, options or {})
        return self

    def scale_current(self, scale, options={}):
        """Scale raw current columns on this CV and return ``self``.
        
        Parameters
        ----------
        scale : float
            Multiplier applied to raw current columns.
        options : dict, optional
            Metadata options. See ``e.describe_options("cv.scale_current")``.
        
        Returns
        -------
        cv
            This mutated CV object.
        
        Examples
        --------
        >>> cv_obj.scale_current(1.25)
        """
        _apply_current_scale(self, scale, options or {})
        return self

    # add unassigned CV-analysis options from dataclass defaults
    def _cv_analysis_options(self, options):
        defaults = PeakCurrentOptions.from_options({}).to_legacy_dict()

        # replace default options with added options
        normalized_options = {
            normalize_key(key).replace("_", " "): value
            for key, value in (options or {}).items()
        }
        defaults.update(normalized_options)
        options = defaults

        r = options["tangent range"]
        if isinstance(r, (int, float, np.number)):  # single numeric → expand to two-element list
            r = [r / 10, r]
        options["tangent range"] = r

        return options

    def analysis_segment_data(self, options):
        options = self._cv_analysis_options(options)
        
        x, y = self.xy(options)

        segments = options.get('segments') or options.get('segment')
        if segments is None:
            return x, y

        return self._select_segments(x, y, segments)

    def find_segment(
            self,
            *,  # force keyword-only for clarity
            idx=None,  # global index of a point in self.x()
            potential=None,  # …or potential value (float)
            x=None,  # full x-array may be passed in to avoid recompute
            options=None
    ):
        """
        Identify the CV segment containing *idx* or *potential*.

        Returns
        -------
        seg_idx      : int               # 0-based segment number
        seg_slice    : slice             # slice object for that segment
        seg_bounds   : tuple(start,end)  # start/end indices (inclusive/exclusive)

        Notes
        -----
        - Exactly one of *idx* or *potential* must be supplied.
        - Supply *x* if you already have it to avoid an extra self.x(...) call.
        - *options* is forwarded to self.get_xy / self.x so you can respect
          'normalize' or other axis-unit choices if you need them later.
        """
        if (idx is None) == (potential is None):
            raise ValueError("Provide **either** idx **or** potential.")

        # Full x-array — take what we’re given or rebuild it once.
        if x is None:
            if options is None:
                x = self.x().values  # fastest path
            else:
                x, _y = self.get_xy(options.get("normalize", False),
                                    options.get("normalize params"))
        # Resolve idx from potential if necessary
        if idx is None:
            idx = int(np.argmin(np.abs(x - potential)))

        # Locate segment boundaries
        dx = np.diff(x)
        break_pts = np.where(np.diff(np.sign(dx)))[0] + 1  # switch in scan direction
        seg_bounds = np.concatenate(([0], break_pts, [len(x)]))

        seg_idx = np.searchsorted(seg_bounds, idx, side="right") - 1
        start, end = seg_bounds[seg_idx], seg_bounds[seg_idx + 1]
        return seg_idx, slice(start, end), (start, end)

    def fft(self, options={}):
        """
        Plots the FFT amplitude spectrum of the CV's current data.

        This method helps visualize the frequency components of the current signal,
        which can be useful for identifying noise characteristics.

        Relevant options (to be managed by _cv_analysis_options or passed directly):
            'fft_plot_title' (str, default: "FFT Amplitude Spectrum for [CV Name]"): Plot title.
            'fft_plot_xlabel' (str, default: "Frequency (Hz)" or "Normalized Frequency"): X-axis label.
            'fft_plot_ylabel' (str, default: "Amplitude"): Y-axis label.
            'fft_ylog_scale' (bool, default: False): Whether to use a log scale for the y-axis.
            # User might need to ensure these options are part of 'cv analysis' defaults
            # or handle them if _cv_analysis_options doesn't cover them.
        Returns:
            tuple: (fig, ax) Matplotlib figure and axes objects of the plot.
                   Returns (None, None) if plotting fails.
        """
        # Process options using the class's existing _cv_analysis_options method
        # Note: self._cv_analysis_options is tailored for 'cv analysis'. You might need to
        # keep FFT-specific options explicit at the call site
        # or handle them locally if they are not part of 'cv analysis'.
        options = self._cv_analysis_options(options)

        plot_title_base = options.get('fft_plot_title_base', "FFT Amplitude Spectrum")
        plot_title = f"{plot_title_base} for {self.name}"
        plot_xlabel = options.get('fft_plot_xlabel', None)  # Default set later based on Fs
        plot_ylabel = options.get('fft_plot_ylabel', "Amplitude")
        ylog_scale = options.get('fft_ylog_scale', False)

        try:
            current_data = self.y(options).values  # Use processed options for y()
        except Exception as e:
            print(f"Error accessing current data for FFT: {e}")
            return None, None

        if len(current_data) == 0:
            print("No current data points to perform FFT.")
            return None, None

        # --- Calculate Sampling Frequency (Fs) ---
        sampling_frequency_Hz = 0
        delta_t_s = 0
        # self.delta_x is the potential step from the echem object, self.scan_rate is V/s
        if hasattr(self, 'scan_rate') and self.scan_rate > 0 and \
                hasattr(self, 'delta_x') and self.delta_x > 0:
            delta_t_s = self.delta_x / self.scan_rate
            if delta_t_s > 0:
                sampling_frequency_Hz = 1.0 / delta_t_s
            else:
                print("Warning: Calculated Delta t for FFT is zero or negative. Using normalized frequency.")
        else:
            print("Warning: Scan rate or delta_x not available/valid. Using normalized frequency for FFT.")

        # --- Perform FFT ---
        N = len(current_data)
        # Remove DC offset for better visualization of AC components
        yf = np.fft.fft(current_data - np.mean(current_data))

        if sampling_frequency_Hz > 0 and N > 0:
            xf = np.fft.fftfreq(N, d=delta_t_s)[:N // 2]  # Frequencies up to Nyquist
            effective_xlabel = plot_xlabel if plot_xlabel else "Frequency (Hz)"
        elif N > 0:  # Fallback to normalized frequency
            xf = np.fft.fftfreq(N, d=1)[:N // 2]  # d=1 for normalized frequency
            effective_xlabel = plot_xlabel if plot_xlabel else "Normalized Frequency (cycles/sample)"
            print("Displaying FFT with normalized frequency (0 to 0.5 cycles/sample).")
        else:  # Should have been caught by len(current_data) == 0
            return None, None

        amplitude_spectrum = 2.0 / N * np.abs(yf[0:N // 2]) if N > 0 else np.array([])

        if N <= 1 or len(xf) == 0 or len(amplitude_spectrum) == 0:
            print("Warning: FFT resulted in empty spectrum arrays. Cannot plot.")
            return None, None

        # --- Plot FFT Spectrum ---
        if options.get('plot'):
            fig, ax = plt.subplots(
                figsize=options.get('figure.figsize', (10, 5)))  # Use existing MPL defaults if possible

            # Exclude DC component (xf[0], amplitude_spectrum[0]) for clearer plotting of AC components
            if len(xf) > 1 and len(amplitude_spectrum) > 1:
                ax.plot(xf[1:], amplitude_spectrum[1:])
            elif len(xf) > 0 and len(amplitude_spectrum) > 0:  # Plot even if only DC (though less useful)
                ax.plot(xf, amplitude_spectrum)

            ax.set_title(plot_title)
            ax.set_xlabel(effective_xlabel)
            ax.set_ylabel(plot_ylabel)

            if ylog_scale:
                ax.set_yscale('log')

            if sampling_frequency_Hz > 0 and len(xf) > 0:
                ax.set_xlim(0, sampling_frequency_Hz / 2)  # Show up to Nyquist frequency
            elif len(xf) > 0:  # Normalized frequency plot
                ax.set_xlim(0, 0.5)  # Normalized frequency up to Nyquist (0.5 cycles/sample)

            plt.tight_layout()
            # The plot will be shown if used in an interactive environment, or can be saved by the user.
            # To explicitly show here: plt.show()

            return fig, ax

        return xf, yf

    def noise_filter(self, options={}):
        cuttoff_Hz = options.get('noise cutoff',9)

    def current_at_potential(self, potential, options={}):
        """
        Returns the current at a specified potential. If no segment(s) are provided,
        reports the current from all segments.

        Parameters:
            potential (float): The potential (in volts) at which to retrieve the current.
            options (dict): Dictionary of options. Can include:
                - 'segment': int
                - 'segments': list of int
                - 'normalize': bool
                - 'normalize params': dict
                - 'plot': bool (default: True)
                - 'print': bool (default: True)

        Returns:
            CVAnalysisResult: dictionary-compatible mapping
            ``{segment_number: (potential, current) or None}`` with ``.table`` and
            ``.primary`` convenience attributes.
        """
        options = self._cv_analysis_options(options)
        norm = options.get("normalize", False)
        norm_params = options.get("normalize params", {})
        do_plot = options.get("plot", True)
        do_print = options.get("print", True)

        # Segment selection logic
        if options.get('segment', None) is not None:
            segments = [options['segment']]
        elif options.get('segments', None) is not None:
            segments = options['segments']
            if isinstance(segments, int):
                segments = [segments]
        else:
            segments = list(range(1, self.segments + 1))

        results = {}
        for seg in segments:
            seg_opts = options.copy()
            seg_opts['segment'] = seg
            x, y = self.analysis_segment_data(seg_opts)
            dx_mean = np.abs(np.mean(np.diff(x)))
            min_distance = np.min(np.abs(x - potential))

            if min_distance > 3 * dx_mean:
                results[seg] = None
                continue

            peak_index = np.argmin(np.abs(x - potential))
            p, i = x[peak_index], y[peak_index]
            results[seg] = (p, i)

            if do_plot:
                plot_keys = {field.name.replace("_", " ") for field in fields(PlotOptions)}
                plot_opts = {
                    key: value
                    for key, value in seg_opts.items()
                    if key in plot_keys
                }
                plot_opts["plot segment"] = seg
                plot_opts.pop("plot segments", None)
                if options.get("plot cv", True):
                    self.plot(plot_opts)
                x_scale, y_scale = self.xy_scale(plot_opts)
                plt.scatter(
                    x[peak_index] * x_scale,
                    y[peak_index] * y_scale + options.get('offset', 0),
                    color='black', zorder=3
                )

        rows = []
        primary = None
        for seg, item in results.items():
            if item is None:
                rows.append({"metric": "Current", "segment": seg, "value": "not found", "kind": "plain"})
                continue
            p, i = item
            if primary is None:
                primary = i
            rows.extend([
                {"metric": "Potential", "segment": seg, "value": p, "kind": "potential"},
                {"metric": "Current", "segment": seg, "value": i, "kind": "current"},
            ])
        result = CVAnalysisResult(
            results,
            primary=primary,
            table=_cv_analysis_table(self, rows, options),
            summary={"analysis": "current_at_potential", "target potential": potential},
            diagnostics={},
            axes=plt.gca() if plt.get_fignums() else None,
        )
        if do_print:
            result.show(options)
        return result

    # get Ep
    def peak_potential_old(self, options={}):
        options = self._cv_analysis_options(options)

        # Smoothing the y data using Savitzky-Golay filter
        temp_options = options.copy()
        temp_options['smooth'] = False
        x, y = self.analysis_segment_data(options)
        smoothed_y, _, _, sg_meta = _savgol_bundle(y, options, delta=self.delta_x)

        if options.get("troubleshoot"):
            print(f"SG window={sg_meta['window']}, polyorder={sg_meta['polyorder']}")

        # determine prominence
        prominence = options.get('peak prominence')
        noise_std_dev = None
        if prominence is None:
            if len(smoothed_y) > 1 and prominence is None:
                current_diffs = np.diff(smoothed_y)
                noise_std_dev = np.std(current_diffs) / np.sqrt(2)
            else:
                noise_std_dev = 0
            prominence = 5 * noise_std_dev

        if options.get('troubleshoot'):
            if noise_std_dev is not None:
                print(f"Estimated noise std dev: {noise_std_dev}")
            print(f"Calculated dynamic prominence: {prominence}")

        # Find peaks in the smoothed y data
        maxima, _ = find_peaks(smoothed_y, prominence = prominence)
        minima, _ = find_peaks(-smoothed_y, prominence = prominence)
        peaks = np.concatenate((maxima, minima))

        # If no peaks are found, return None
        if len(peaks) == 0:
            selection = 'CV trace'
            if options.get('segments') or options.get('segment'):
                selection = 'selected segment(s)'
            raise ValueError(
                f"peak_potential could not locate any peaks in the {selection}. "
                "Check the 'guess potential', 'peak prominence' threshold, and 'noise window' options, "
                "or verify that the segment actually contains a local extremum. "
                "Use the 'troubleshoot' option for more help."
            )

        exact_potential = options.get('exact potential')
        guess_potential = options.get('guess potential')

        if exact_potential is not None:
            # Find index closest to the exact potential (no peak logic)
            peak_index = int(np.argmin(np.abs(x - exact_potential)))

        elif guess_potential is not None:
            # Find the peak potential (x-value) closest to the guessed value
            peak_index = peaks[np.argmin(np.abs(x[peaks] - guess_potential))]

        else:
            # Default: largest absolute peak
            peak_index = peaks[np.argmax(np.abs(smoothed_y[peaks]))]

        peak_potential = round_sigfigs(x[peak_index], options["sig figs"])

        if options["print"]:
            x_name = self.x(options).name
            x_unit = self.units.get(x_name, '')
            print(f"Ep: {peak_potential} {x_unit}".strip())
        if options["plot"]:
            if not options.get('internal call'):
                self._plot_from_analysis_options(options)
            x_scale, y_scale = self.xy_scale(options)
            plt.scatter(
                x[peak_index] * x_scale,
                y[peak_index] * y_scale + options.get('offset', 0),
                color='tab:blue',zorder=3
            )
            if options.get('troubleshoot'):
                plt.scatter(
                    x[peaks] * x_scale,
                    y[peaks] * y_scale + options.get('offset', 0),
                    color='tab:blue', zorder=3, s=10
                )
            
        return {
            "Ep": peak_potential,
            "index": peak_index,
            "current": y[peak_index],
        }

    def peak_potential(self, options={}):
        """Find the peak potential for a selected CV segment.
        
        Parameters
        ----------
        options : dict or PeakPotentialOptions, optional
            Segment, smoothing, derivative, and peak-selection options. See ``e.describe_options("cv.peak_potential")``.
        
        Returns
        -------
        CVAnalysisResult
            Dictionary-compatible result with ``Ep``, row index, and a tidy display table.
        
        Examples
        --------
        >>> peak = cv_obj.peak_potential({"guess potential": -1.5, "segment": 1})
        >>> peak["Ep"]
        """
        typed_options = PeakPotentialOptions.from_options(options)
        options = self._cv_analysis_options(typed_options.to_legacy_dict())

        temp_options = options.copy()
        temp_options["smooth"] = False
        x, y = self.analysis_segment_data(temp_options)

        exact_potential = options.get("exact potential")
        guess_potential = options.get("guess potential")

        # exact-potential mode should bypass extrema finding entirely
        if exact_potential is not None:
            peak_index = int(np.argmin(np.abs(x - float(exact_potential))))
            peak_potential = round_sigfigs(x[peak_index], options["sig figs"])

            if options["plot"]:
                if not options.get("internal call") and options.get("plot cv", True):
                    self._plot_from_analysis_options(options)

                x_scale, y_scale = self.xy_scale(options)

                plt.scatter(
                    x[peak_index] * x_scale,
                    y[peak_index] * y_scale + options.get("offset", 0),
                    color="tab:blue",
                    zorder=3,
                )

                if options.get("troubleshoot"):
                    print("Using 'exact potential'; extrema detection was bypassed.")

            values = {
                "Ep": peak_potential,
                "index": peak_index,
                "current": y[peak_index],
            }
            result = _cv_analysis_result(
                self,
                "peak_potential",
                values,
                [{"metric": "Ep", "value": peak_potential, "kind": "potential"}],
                "Ep",
                options,
                diagnostics={"index": peak_index, "current": y[peak_index]},
            )
            if options["print"]:
                result.show(options)
            return result

        extrema, smoothed_y, prom_map, ext_meta = _find_extrema_indices(
            y,
            options,
        )

        if len(extrema) == 0:
            selection = "CV trace"
            if options.get("segments") or options.get("segment"):
                selection = "selected segment(s)"
            raise ValueError(
                f"peak_potential could not locate any extrema in the {selection}. "
                "Check 'guess potential', 'peak prominence', or smoothing settings."
            )

        if guess_potential is not None:
            peak_index = int(extrema[np.argmin(np.abs(x[extrema] - guess_potential))])
        else:
            # Sign-agnostic: choose the most prominent extremum, not largest |current|.
            peak_index = max(extrema, key=lambda idx: prom_map.get(int(idx), 0.0))

        peak_potential = round_sigfigs(x[peak_index], options["sig figs"])

        if options["plot"]:
            if not options.get("internal call") and options.get("plot cv", True):
                self._plot_from_analysis_options(options)

            x_scale, y_scale = self.xy_scale(options)

            plt.scatter(
                x[peak_index] * x_scale,
                y[peak_index] * y_scale + options.get("offset", 0),
                color="tab:blue",
                zorder=3,
            )

            if options.get("troubleshoot"):
                plt.scatter(
                    x[extrema] * x_scale,
                    y[extrema] * y_scale + options.get("offset", 0),
                    color="tab:blue",
                    s=10,
                    zorder=3,
                )
                print(
                    f"SG window={ext_meta['sg window']}, "
                    f"polyorder={ext_meta['sg polyorder']}, "
                    f"prominence={ext_meta['prominence']}"
                )

        values = {
            "Ep": peak_potential,
            "index": peak_index,
            "current": y[peak_index],
        }
        result = _cv_analysis_result(
            self,
            "peak_potential",
            values,
            [{"metric": "Ep", "value": peak_potential, "kind": "potential"}],
            "Ep",
            options,
            diagnostics={"index": peak_index, "current": y[peak_index]},
        )
        if options["print"]:
            result.show(options)
        return result

    def _resolve_auto_tangent_mask(
            self,
            x_pre,
            y_pre,
            v_pre,
            a_pre,
            options,
            min_pts,
            target_extremum_kind=None,
    ):
        """
        Resolve the automatic tangent-fit search mask.

        The algorithm works in scan-order coordinates, not visual left/right
        plot coordinates:

        1. Determine an end boundary from a target-local |dI/dE| peak.
        2. Find curvature-relevant current extrema before that end boundary.
        3. Try each extrema as a candidate start boundary, newest to oldest.
        4. Accept the latest start boundary that yields enough quiet points.
        5. If no extrema candidate works, try the beginning of the pre-target region.

        Quiet points are defined as points with low absolute first derivative
        and low absolute second derivative relative to the candidate domain.
        """
        activity = np.abs(v_pre)
        activity_max = np.nanmax(activity)

        if not np.isfinite(activity_max) or activity_max == 0:
            raise ValueError(
                "Could not determine automatic tangent window because |dI/dE| is flat."
            )

        # ----- end boundary from target-local |dI/dE| peak -----
        # Prefer new scan-order terminology, but keep the old option as an alias.
        tail_fraction = options.get(
            "tangent end peak tail fraction",
            options.get("tangent right peak tail fraction", 0.30),
        )

        tail_start = max(0, int((1 - tail_fraction) * len(activity)))
        activity_tail = activity[tail_start:]

        prom_pct = options.get("tangent peak prominence percentile", 85)
        tail_max = np.nanmax(activity_tail) if len(activity_tail) > 0 else activity_max
        prominence = max(np.percentile(activity_tail, prom_pct), 0.03 * tail_max)

        peak_distance = options.get("tangent peak distance")
        if peak_distance is None:
            peak_distance = max(5, int(0.08 * len(x_pre)))

        peaks, props = find_peaks(
            activity,
            prominence=prominence,
            distance=peak_distance,
        )

        peaks_in_tail = peaks[peaks >= tail_start]

        if len(peaks_in_tail) > 0:
            end_idx = int(peaks_in_tail[-1])
            end_mode = "last prominent tail activity peak"
        else:
            if tail_start >= len(activity):
                tail_start = max(0, len(activity) - max(min_pts, 5))

            end_idx = int(tail_start + np.argmax(activity[tail_start:]))
            end_mode = "tail max fallback"

        # ----- candidate start boundaries from relevant y-extrema before end -----
        extrema, y_smooth, prom_map, ext_meta = _find_extrema_indices(
            y_pre[:end_idx],
            options,
        )

        # Avoid using extrema too close to the end boundary. If an extrema is
        # basically part of the same wave onset, it should not define the start.
        extrema = extrema[extrema < end_idx - min_pts]

        relevant_ext, curv_meta = _filter_extrema_by_curvature(
            extrema,
            a_pre[:end_idx],
            options,
        )

        # Try newest same-kind relevant extrema first, then backtrack earlier.
        # "Same-kind" means max-before-max or min-before-min relative to the
        # target peak. If target kind cannot be determined, fall back to the
        # previous any-extrema behavior.
        extrema_kind_map = ext_meta.get("extrema kind map", {})

        start_extrema_kind = str(
            options.get("tangent start extrema kind", "same")
        ).lower().replace("_", " ").replace("-", " ")

        if (
                start_extrema_kind in ("same", "same kind", "same-kind")
                and target_extremum_kind in ("max", "min")
        ):
            candidate_extrema = [
                int(idx)
                for idx in relevant_ext
                if extrema_kind_map.get(int(idx)) == target_extremum_kind
            ]
            candidate_source = "same-kind curvature-relevant extrema"

        elif start_extrema_kind in ("max", "maximum"):
            candidate_extrema = [
                int(idx)
                for idx in relevant_ext
                if extrema_kind_map.get(int(idx)) == "max"
            ]
            candidate_source = "max curvature-relevant extrema"

        elif start_extrema_kind in ("min", "minimum"):
            candidate_extrema = [
                int(idx)
                for idx in relevant_ext
                if extrema_kind_map.get(int(idx)) == "min"
            ]
            candidate_source = "min curvature-relevant extrema"

        else:
            candidate_extrema = [int(idx) for idx in relevant_ext]
            candidate_source = "any curvature-relevant extrema"

        # Backtrack in scan order: newest candidate first, then older candidates.
        start_candidates = list(candidate_extrema[::-1])
        start_candidates.append(0)

        frac_initial = options.get("tangent activity fraction", 0.20)
        frac_sequence = [frac_initial]

        for frac_try in [0.25, 0.30, 0.40]:
            if frac_try > frac_initial and frac_try not in frac_sequence:
                frac_sequence.append(frac_try)

        best = None
        failed_candidates = []

        indices = np.arange(len(x_pre))

        for candidate_rank, start_idx_try in enumerate(start_candidates, start=1):
            domain = (
                    (indices > start_idx_try) &
                    (indices < end_idx)
            )

            domain_count = int(np.count_nonzero(domain))

            if domain_count < min_pts:
                failed_candidates.append({
                    "candidate rank": candidate_rank,
                    "start index": int(start_idx_try),
                    "start potential": float(x_pre[start_idx_try]),
                    "domain points": domain_count,
                    "reason": "domain has fewer than min_pts",
                })
                continue

            # Recompute references for this candidate domain.
            v_ref = np.nanmax(np.abs(v_pre[domain]))
            a_ref = np.nanmax(np.abs(a_pre[domain]))

            if not np.isfinite(v_ref) or v_ref == 0:
                v_ref = 1.0
            if not np.isfinite(a_ref) or a_ref == 0:
                a_ref = 1.0

            for frac in frac_sequence:
                mask = (
                        domain &
                        (np.abs(v_pre) <= frac * v_ref) &
                        (np.abs(a_pre) <= frac * a_ref)
                )

                num_points = int(np.count_nonzero(mask))

                if num_points >= min_pts:
                    if start_idx_try == 0:
                        start_mode = "pre-target start fallback"
                    elif candidate_rank == 1:
                        start_mode = f"latest viable {candidate_source}"
                    else:
                        start_mode = f"backtracked {candidate_source}"

                    best = {
                        "mask": mask,
                        "start index": int(start_idx_try),
                        "start mode": start_mode,
                        "candidate rank": int(candidate_rank),
                        "fraction": float(frac),
                        "v ref": float(v_ref),
                        "a ref": float(a_ref),
                        "v cutoff": float(frac * v_ref),
                        "a cutoff": float(frac * a_ref),
                        "num points": num_points,
                        "domain points": domain_count,
                    }
                    break

            if best is not None:
                break

            failed_candidates.append({
                "candidate rank": candidate_rank,
                "start index": int(start_idx_try),
                "start potential": float(x_pre[start_idx_try]),
                "domain points": domain_count,
                "reason": "not enough low-activity points",
            })

        if best is None:
            raise ValueError(
                "Could not identify enough low-activity tangent points inside the auto domain. "
                "Try setting 'tangent range' or 'tangent potential' manually."
            )

        mask = best["mask"]
        start_idx = best["start index"]

        return mask, {
            "mode": "auto tangent: latest viable extrema-bounded low-activity region",

            # start boundary
            "start mode": best["start mode"],
            "start index": int(start_idx),
            "start potential": float(x_pre[start_idx]),
            "start candidate rank": int(best["candidate rank"]),

            # end boundary
            "end mode": end_mode,
            "end index": int(end_idx),
            "end potential": float(x_pre[end_idx]),

            # extrema / boundary diagnostics
            "target extremum kind": target_extremum_kind,
            "start extrema kind option": start_extrema_kind,
            "start candidate source": candidate_source,
            "y extrema": extrema.tolist(),
            "relevant y extrema": relevant_ext.tolist(),
            "extrema kind map": extrema_kind_map,
            "curvature meta": curv_meta,

            # low-activity mask diagnostics
            "fraction": float(best["fraction"]),
            "v ref": float(best["v ref"]),
            "a ref": float(best["a ref"]),
            "v cutoff": float(best["v cutoff"]),
            "a cutoff": float(best["a cutoff"]),
            "num points": int(best["num points"]),
            "domain points": int(best["domain points"]),
            "failed start candidates": failed_candidates,
        }

    def _plot_tangent_troubleshoot(
            self,
            x_seg,
            y_seg,
            idx_local_target,
            range_mask,
            pre_mask,
            tan_idx_local,
            options,
            auto_meta=None,
    ):
        """
        Plot full-segment current, velocity, and acceleration for tangent troubleshooting.

        Highlights:
        - pre-target region
        - auto tangent domain (range_mask over x_pre)
        - final tangent-fit block
        """
        offset = options.get("offset", 0)

        _, v_seg, a_seg, sg_meta = _savgol_bundle(
            y_seg,
            options,
            delta=self.delta_x,
        )

        if auto_meta is not None:
            print("Auto tangent metadata:", auto_meta)
        print(f"SG window={sg_meta['window']}, polyorder={sg_meta['polyorder']}")

        x_scale, y_scale = self.xy_scale(options)

        # Scale all derivative traces to the full displayed current magnitude
        y_display_seg = y_seg * y_scale
        v_display_seg = _scale_trace_to_match_current(v_seg, y_display_seg)
        a_display_seg = _scale_trace_to_match_current(a_seg, y_display_seg)

        x_plot = x_seg * x_scale
        y_plot = y_display_seg + offset
        v_plot = v_display_seg + offset
        a_plot = a_display_seg + offset

        ax = plt.gca()

        # Plot full-segment traces
        plt.plot(x_plot, y_plot, color="black", alpha=0.6, label="current")
        plt.plot(x_plot, v_plot, color="tab:blue", alpha=0.9, label="v")
        plt.plot(x_plot, a_plot, color="tab:orange", alpha=0.9, label="a")

        # Pre-target region in segment coordinates
        pre_idx_seg = np.flatnonzero(pre_mask)
        if len(pre_idx_seg) > 0:
            x_pre_seg = x_seg[pre_idx_seg]
            plt.axvspan(
                x_pre_seg[0] * x_scale,
                x_pre_seg[-1] * x_scale,
                color="grey",
                alpha=0.08,
                label="pre-target region",
            )

        # Search window in segment coordinates
        range_idx_pre = np.flatnonzero(range_mask)
        if len(range_idx_pre) > 0:
            x_pre = x_seg[pre_mask]
            x_search = x_pre[range_idx_pre]

            plt.axvspan(
                x_search[0] * x_scale,
                x_search[-1] * x_scale,
                color="gold",
                alpha=0.20,
                label="auto tangent domain",
            )

            # Final tangent-fit block within search window
            if tan_idx_local is not None and len(tan_idx_local) > 0:
                x_fit = x_search[tan_idx_local]
                plt.axvspan(
                    x_fit[0] * x_scale,
                    x_fit[-1] * x_scale,
                    color="tab:red",
                    alpha=0.18,
                    label="fit block",
                )
                plt.scatter(
                    x_fit * x_scale,
                    y_seg[pre_mask][range_mask][tan_idx_local] * y_scale + offset,
                    s=12,
                    color="tab:red",
                    zorder=4,
                )

        # Target peak position
        plt.axvline(
            x_seg[idx_local_target] * x_scale,
            color="tab:green",
            linestyle="--",
            alpha=0.8,
            label="target peak",
        )

        if auto_meta is not None:
            print("Auto tangent metadata:", auto_meta)

        plt.legend()

    def _fit_tangent_line(self, x, y, idx_target=None, tangent_potential=None, options=None):
        if options is None:
            options = {}

        if tangent_potential is None and idx_target is None:
            raise ValueError("Provide either idx_target or tangent_potential.")

        # -------- automatic tangent based on target index --------
        if tangent_potential is None:
            E_target = x[idx_target]

            # 1. Identify which scan segment contains the target point
            seg_idx, seg_slice, _ = self.find_segment(idx=idx_target, x=x)

            # 2. Clip x & y to one segment, then to pre-target data
            x_seg = x[seg_slice]
            y_seg = y[seg_slice]
            idx_local_target = idx_target - seg_slice.start

            pre_mask = np.arange(len(x_seg)) < idx_local_target
            x_pre = x_seg[pre_mask]
            y_pre = y_seg[pre_mask]

            if len(x_pre) < 5:
                raise ValueError("Not enough pre-peak points to fit a tangent line.")

            # 3. Velocity and acceleration
            _, v_pre, a_pre, sg_meta = _savgol_bundle(
                y_pre,
                options,
                delta=self.delta_x,
            )

            y_seg_smooth, _ = _savgol_apply(y_pre, options, deriv=0)
            target_extremum_kind = _classify_extremum_kind(
                y_seg_smooth,
                idx_local_target,
            )

            # 4. Optional potential window
            tangent_range = options.get("tangent range", "auto")

            min_pts = max(5, int(0.5 * len(x_pre) ** 0.5))

            if tangent_range in (None, "auto"):
                range_mask, auto_meta = self._resolve_auto_tangent_mask(
                    x_pre=x_pre,
                    y_pre=y_pre,
                    v_pre=v_pre,
                    a_pre=a_pre,
                    options=options,
                    min_pts=min_pts,
                    target_extremum_kind=target_extremum_kind,
                )
            else:
                if idx_target == 0:
                    direction = 1
                else:
                    direction = 1 if x[idx_target] - x[idx_target - 1] > 0 else -1

                r0, r1 = tangent_range
                range_mask = (
                        (x_pre * direction < E_target * direction - r0) &
                        (x_pre * direction > E_target * direction - r1)
                )

                if not np.any(range_mask):
                    raise ValueError(
                        "No data points lie within the requested 'tangent range' window "
                        f"{tangent_range}."
                    )

            x_range = x_pre[range_mask]
            y_range = y_pre[range_mask]
            v_range = v_pre[range_mask]
            a_range = a_pre[range_mask]

            if len(x_range) < 5:
                raise ValueError("Not enough points remain in the tangent-fit window.")

            # 5. Choose tangent points
            min_pts = max(5, int(0.5 * len(x_range) ** 0.5))
            tangent_mask = None

            if options.get("percent threshold") is not None:
                percentile_sequence = [options["percent threshold"]]
            else:
                percentile_sequence = [10, 20, 30, 40, 50]

            tan_idx_local = None

            for pct in percentile_sequence:
                v_thr = np.percentile(np.abs(v_range), pct)
                a_thr = np.percentile(np.abs(a_range), pct)

                fit_mask = (np.abs(v_range) <= v_thr) & (np.abs(a_range) <= a_thr)

                if np.count_nonzero(fit_mask) >= min_pts:
                    tan_idx_local = np.flatnonzero(fit_mask)
                    if options.get("troubleshoot"):
                        print(f"{pct}% worked.")
                    break
                elif options.get("troubleshoot"):
                    print(f"{pct}% did not work.")

            if tan_idx_local is None:
                raise ValueError(
                    "Could not find enough tangent-fit points. "
                    "Try increasing 'percent threshold' or setting 'tangent range' manually."
                )

            if options.get("troubleshoot", False):
                self._plot_tangent_troubleshoot(
                    x_seg=x_seg,
                    y_seg=y_seg,
                    idx_local_target=idx_local_target,
                    range_mask=range_mask,
                    pre_mask=pre_mask,
                    tan_idx_local=tan_idx_local,
                    options=options,
                    auto_meta=auto_meta if tangent_range in (None, "auto") else None,
                )

                x_pre = x_seg[pre_mask]
                x_search = x_pre[range_mask]
                print(
                    "Tangent fit window:",
                    f"{x_search[tan_idx_local[0]]:.4f} to {x_search[tan_idx_local[-1]]:.4f} V"
                )

            # 6. Linear fit
            m, b = np.polyfit(x_range[tan_idx_local], y_range[tan_idx_local], 1)

            # 7. Map selected tangent indices back to global indices
            range_global = np.flatnonzero(range_mask)
            fit_idx_pre = range_global[tan_idx_local]
            fit_idx_seg = np.flatnonzero(pre_mask)[fit_idx_pre]
            fit_indices = fit_idx_seg + seg_slice.start
            tanline_start = int(np.min(fit_indices))

            if options.get("troubleshoot", False):
                x_scale, y_scale = self.xy_scale(options)
                plt.scatter(
                    x[fit_indices] * x_scale,
                    y[fit_indices] * y_scale + options.get("offset", 0),
                    s=10,
                    color="tab:red",
                    zorder=3,
                )

        # -------- user-provided tangent potential --------
        else:
            E_tan = tangent_potential
            E_tan_idx = int(np.argmin(np.abs(x - E_tan)))

            seg_idx, seg_slice, _ = self.find_segment(idx=E_tan_idx, x=x)
            x_seg = x[seg_slice]
            y_seg = y[seg_slice]

            min_pts = max(5, int(0.5 * len(x_seg) ** 0.5))
            order = np.argsort(np.abs(x_seg - E_tan))
            best = np.sort(order[:min_pts])

            m, b = np.polyfit(x_seg[best], y_seg[best], 1)

            fit_indices = seg_slice.start + best
            tanline_start = int(np.min(fit_indices))

            if options.get("plot all"):
                x_scale, y_scale = self.xy_scale(options)
                plt.scatter(
                    x[fit_indices] * x_scale,
                    (m * x[fit_indices] + b) * y_scale + options.get("offset", 0),
                    s=10,
                    color="tab:red",
                    zorder=3,
                )

        return {
            "slope": m,
            "intercept": b,
            "tanline_start": tanline_start,
            "fit_indices": fit_indices,
            "segment_slice": seg_slice,
        }

    def peak_current(self, options={}):
        """Measure peak current using a tangent-background correction.
        
        Parameters
        ----------
        options : dict or PeakCurrentOptions, optional
            Peak-potential, tangent-line, plot, and print options. See ``e.describe_options("cv.peak_current")``.
        
        Returns
        -------
        CVAnalysisResult
            Dictionary-compatible result with ``ip``, ``Ep``, tangent diagnostics, and a tidy display table.
        
        Examples
        --------
        >>> ip, *_ = cv_obj.peak_current({"guess potential": -1.5, "segment": 1})
        """
        typed_options = PeakCurrentOptions.from_options(options)
        options = self._cv_analysis_options(typed_options.to_legacy_dict())

        if options["plot"] and not options.get("internal call") and options.get("plot cv", True):
            self._plot_from_analysis_options(options)
            options["new plot"] = False
        
        def _peak_current_fallback_mode(value):
            if value is None:
                return "none"
            mode = str(value).strip().lower().replace("_", " ").replace("-", " ")
            if mode in {"none", "error", "raise"}:
                return "none"
            if mode in {"guess potential", "exact potential"}:
                return "guess potential"
            return "highest current"

        # confirm peak potential
        internal_peak_options = typed_options.for_peak_potential()
        internal_peak_options = replace(
            internal_peak_options,
            plot=typed_options.plot and typed_options.plot_peak_potential,
            print=typed_options.print_all,
            internal_call=True,
            new_plot=False,
        )
        peak_source = "peak"
        peak_potential_error = None
        try:
            peak_result = self.peak_potential(internal_peak_options)
            E_peak = peak_result["Ep"]
            idx_E_peak = peak_result["index"]
        except ValueError as exc:
            message = str(exc)
            if "could not locate any extrema" not in message and "could not locate any peaks" not in message:
                raise

            fallback_mode = _peak_current_fallback_mode(options.get("peak fallback"))
            if fallback_mode == "none":
                raise

            peak_potential_error = message
            if fallback_mode == "guess potential":
                guess_potential = options.get("guess potential")
                if guess_potential is None:
                    raise ValueError(
                        "'peak fallback'='guess potential' requires 'guess potential'."
                    ) from exc
                fallback_peak_options = replace(
                    internal_peak_options,
                    exact_potential=guess_potential,
                )
                peak_result = self.peak_potential(fallback_peak_options)
                E_peak = peak_result["Ep"]
                idx_E_peak = peak_result["index"]
                peak_source = "guess potential fallback"
            else:
                x_fallback, y_fallback = self.analysis_segment_data(options)
                if len(y_fallback) == 0:
                    raise ValueError(
                        "peak_current could not use the highest-current fallback because "
                        "the selected segment contains no data."
                    ) from exc
                idx_E_peak = int(np.nanargmax(np.abs(y_fallback)))
                E_peak = round_sigfigs(x_fallback[idx_E_peak], options["sig figs"])
                peak_result = {
                    "Ep": E_peak,
                    "index": idx_E_peak,
                    "current": y_fallback[idx_E_peak],
                }
                peak_source = "highest current fallback"

                if options["plot"] and typed_options.plot_peak_potential:
                    x_scale, y_scale = self.xy_scale(options)
                    plt.scatter(
                        x_fallback[idx_E_peak] * x_scale,
                        y_fallback[idx_E_peak] * y_scale + options.get("offset", 0),
                        color="tab:blue",
                        zorder=3,
                    )

        x, y = self.analysis_segment_data(options)

        tan = self._fit_tangent_line(
            x, y,
            idx_target=idx_E_peak,
            tangent_potential=options.get("tangent potential"),
            options=options
        )

        m = tan["slope"]
        b = tan["intercept"]
        tanline_start = tan["tanline_start"]
        tanline = [m, b]

        base_current = m * E_peak + b
        peak_current = round_sigfigs(y[idx_E_peak] - base_current, options["sig figs"])

        if options["plot"]:
            x_scale, y_scale = self.xy_scale(options)

            i0, i1 = sorted([tanline_start, idx_E_peak])
            x_tangent = np.array(x[i0:i1 + 1]) * x_scale
            y_tangent = (m * (x_tangent / x_scale) + b) * y_scale

            plt.plot(
                x_tangent,
                y_tangent + options.get("offset", 0),
                linestyle="--",
                color="tab:red",
            )

            x_peak_plot = E_peak * x_scale
            y_min = base_current * y_scale
            y_max = y[idx_E_peak] * y_scale

            plt.vlines(
                x_peak_plot,
                y_min + options.get("offset", 0),
                y_max + options.get("offset", 0),
                color="tab:red",
                linestyle="--",
            )

            if options.get("plot all") and "fit indices" in tan:
                fit_idx = np.asarray(tan["fit indices"], dtype=int)
                plt.scatter(
                    x[fit_idx] * x_scale,
                    y[fit_idx] * y_scale + options.get("offset", 0),
                    s=10,
                    color="tab:red",
                    zorder=3,
                )

        values = {
            "ip": peak_current,
            "Ep": E_peak,
            "Ep index": idx_E_peak,
            "baseline current": base_current,
            "tangent line": tanline,
            "tangent slope": m,
            "tangent intercept": b,
            "tangent start": tanline_start,
            "fit indices": tan.get("fit_indices"),
            "segment slice": tan.get("segment_slice"),
            "peak source": peak_source,
        }
        if peak_potential_error is not None:
            values["peak potential error"] = peak_potential_error
        result = _cv_analysis_result(
            self,
            "peak_current",
            values,
            [
                {"metric": "Ep", "value": E_peak, "kind": "potential"},
                {"metric": "ip", "value": peak_current, "kind": "current"},
            ],
            "ip",
            options,
            diagnostics={
                "Ep index": idx_E_peak,
                "baseline current": base_current,
                "tangent line": tanline,
                "tangent slope": m,
                "tangent intercept": b,
                "tangent start": tanline_start,
                "fit indices": tan.get("fit_indices"),
                "segment slice": tan.get("segment_slice"),
                "peak source": peak_source,
                "peak potential error": peak_potential_error,
            },
        )
        if options["print"]:
            result.show(options)
        return result

    def plateau_current(self, options={}):
        """Analyze plateau current for this CV.

        Parameters
        ----------
        options : dict or PlateauCurrentOptions, optional
            Plateau, normalization, print, and plot options. See ``e.describe_options("cv.plateau_current")``.

        Returns
        -------
        pandas.DataFrame
            Plateau-current summary table.

        Examples
        --------
        >>> result = cv_obj.plateau_current({"non-catalytic cv": blank_cv, "guess potential": -1.5})
        """
        return plateau_current(self, options)

    def half_peak_potential(self, options={}):
        """Estimate the half-peak potential for a selected CV wave.
        
        Parameters
        ----------
        options : dict or PeakPotentialOptions, optional
            Segment, peak-picking, print, and plot options. See ``e.describe_options("cv.half_peak_potential")``.
        
        Returns
        -------
        CVAnalysisResult
            Dictionary-compatible result with ``Ep/2``, ``Δ(Ep - Ep/2)``, and a tidy display table.
        
        Examples
        --------
        >>> E_half_peak = cv_obj.half_peak_potential({"guess potential": -1.5})
        """
        typed_options = PeakCurrentOptions.from_options(options)
        options = self._cv_analysis_options(typed_options.to_legacy_dict())

        if options["plot"] and not options.get('internal call') and options.get("plot cv", True):
            self._plot_from_analysis_options(options)
            options["new plot"] = False
        
        # confirm peak potential and find peak current
        internal_peak_options = typed_options.for_peak_potential()
        internal_peak_options = replace(
            internal_peak_options,
            print=typed_options.print_all,
            internal_call=True,
            new_plot=False,
        )
        internal_current_options = replace(
            typed_options,
            print=typed_options.print_all,
            internal_call=True,
            new_plot=False,
        )
        peak_result = self.peak_potential(internal_peak_options)
        E_peak = peak_result["Ep"]
        idx_E_peak = peak_result["index"]
        current_result = self.peak_current(internal_current_options)
        peak_current = current_result["ip"]
        m, b = current_result["tangent line"]
        tanline_start = current_result["tangent start"]

        # Smoothing the y data using Savitzky-Golay filter
        x, y = self.analysis_segment_data(options)
        y_tan = m * x + b

        # Search window :  tanline_start  →  idx_E_peak
        idx_window = slice(tanline_start, idx_E_peak + 1)
        x_win = x[idx_window]
        y_win = y[idx_window]
        ytan_win = y_tan[idx_window]

        # find Ep/2
        half_current = peak_current / 2
        diff = np.abs(y_win - ytan_win - half_current)
        local_idx = np.argmin(diff)

        # convert back to global index & potential
        idx_half_peak = (np.arange(len(x))[idx_window])[local_idx]
        E_half_peak = round_sigfigs(x[idx_half_peak], options["sig figs"])
        delta = round_sigfigs(E_peak - E_half_peak, options["sig figs"])

        if options["plot"]:
            x_scale, y_scale = self.xy_scale(options)
            plt.hlines(
                y_win[local_idx] * y_scale + options.get('offset', 0),
                E_half_peak * x_scale,
                E_peak * x_scale,
                color='tab:red',
                linestyle="--",
            )

        values = {
            "Ep/2": E_half_peak,
            "Δ(Ep - Ep/2)": delta,
            "delta": delta,
            "Ep": E_peak,
            "Ep index": idx_E_peak,
            "ip": peak_current,
        }
        result = _cv_analysis_result(
            self,
            "half_peak_potential",
            values,
            [
                {"metric": "Ep/2", "value": E_half_peak, "kind": "potential"},
                {"metric": "Δ(Ep - Ep/2)", "value": delta, "kind": "potential"},
                {"metric": "Ep", "value": E_peak, "kind": "potential"},
                {"metric": "ip", "value": peak_current, "kind": "current"},
            ],
            "Ep/2",
            options,
            diagnostics={"Ep index": idx_E_peak},
        )
        if options["print"]:
            result.show(options)
        return result

    def peak_info(self, options={}): ### Change this to have a default plot and print option? Make consistent with wave_info
        options = self._cv_analysis_options(options)
        do_print = options.get('print', True)

        if options.get('plot') and not options.get('internal call') and options.get("plot cv", True):
            self._plot_from_analysis_options(options)
            options["new plot"] = False

        # save print and plot options
        internal_options = options.copy()
        internal_options['print'] = options['print all']
        internal_options['internal call'] = True
        internal_options['new plot'] = False

        # confirm peak potential and find peak current for half wave 1
        x, y = self.analysis_segment_data(options)

        peak_potential_options = {}
        for field in fields(PeakPotentialOptions):
            option_key = field.name.replace("_", " ")
            if option_key in internal_options:
                peak_potential_options[option_key] = internal_options[option_key]
            elif field.name in internal_options:
                peak_potential_options[field.name] = internal_options[field.name]
        peak_result = self.peak_potential(PeakPotentialOptions.from_options(peak_potential_options))
        E_peak = peak_result["Ep"]
        idx_E_peak = peak_result["index"]
        current_result = self.peak_current(internal_options)
        peak_current = current_result["ip"]
        tanline = current_result["tangent line"]
        half_peak_result = self.half_peak_potential(internal_options)
        E_half_peak = half_peak_result["Ep/2"]
        delta = half_peak_result["Δ(Ep - Ep/2)"]
        y_E_peak = y[idx_E_peak]

        peak_info = {
            "Ep": E_peak,
            "ip": peak_current,
            "Ep/2": E_half_peak,
            "Δ(Ep - Ep/2)":delta
        }
        extra_info = {
            "Ep idx": idx_E_peak,
            "tanline": tanline,
            "Ep y":y_E_peak
        }

        values = {
            **peak_info,
            **extra_info,
        }
        result = _cv_analysis_result(
            self,
            "peak_info",
            values,
            [
                {"metric": "Ep", "value": E_peak, "kind": "potential"},
                {"metric": "ip", "value": peak_current, "kind": "current"},
                {"metric": "Ep/2", "value": E_half_peak, "kind": "potential"},
                {"metric": "Δ(Ep - Ep/2)", "value": delta, "kind": "potential"},
            ],
            "Ep",
            options,
            diagnostics=extra_info,
        )
        if do_print:
            result.show(options)
        return result

    def half_wave_potential(self, options={}):
        """Estimate the half-wave potential for a selected CV wave.
        
        Parameters
        ----------
        options : dict or PeakCurrentOptions, optional
            Peak-current, tangent, wave, print, and plot options. See ``e.describe_options("cv.half_wave_potential")``.
        
        Returns
        -------
        CVAnalysisResult
            Dictionary-compatible result with ``E(1/2)``, paired peak data, and a tidy display table.
        
        Examples
        --------
        >>> E_half_wave = cv_obj.half_wave_potential({"guess potential": -1.5})
        """
        options = self._cv_analysis_options(options)

        # save print and plot options
        internal_options = options.copy()
        internal_options['print'] = options.get('print all', False)
        internal_options['plot'] = options.get('plot all', False)
        internal_options['internal call'] = True
        internal_options['new plot'] = False

        segs = options.get("segments", None)
        if segs is None:
            seg1 = options.get("segment", 1)
            if seg1 is None:
                seg1 = 1
            seg2 = seg1 + 1
        else:
            if isinstance(segs, int):
                seg1, seg2 = segs, segs + 1
            elif len(segs) == 2:
                seg1, seg2 = segs
            else:
                raise ValueError(
                    "'segments' for half_wave_potential must be an int or a 2-element sequence."
                )

        guess = options.get('guess potential', None)
        if isinstance(guess, (tuple, list)) and len(guess) == 2:
            guess1, guess2 = guess
        else:
            guess1 = guess
            guess2 = None

        def peak_potential_child_options(source):
            routed = {}
            for field in fields(PeakPotentialOptions):
                option_key = field.name.replace("_", " ")
                if option_key in source:
                    routed[option_key] = source[option_key]
                elif field.name in source:
                    routed[field.name] = source[field.name]
            if source.get("internal call"):
                routed["plot"] = False
            return PeakPotentialOptions.from_options(routed)

        # -------- Peak 1 --------
        seg1_options = internal_options.copy()
        seg1_options.pop("segments", None)
        seg1_options["segment"] = seg1
        if guess1 is not None:
            seg1_options["guess potential"] = guess1

        peak1_result = self.peak_potential(peak_potential_child_options(seg1_options))
        E_peak1 = peak1_result["Ep"]
        idx1 = peak1_result["index"]
        x1, y1 = self.analysis_segment_data(seg1_options)
        y_E_peak1 = y1[idx1]

        # -------- Peak 2 --------
        seg2_options = internal_options.copy()
        seg2_options.pop("segments", None)
        seg2_options["segment"] = seg2
        if guess2 is None:
            guess2 = E_peak1
        seg2_options["guess potential"] = guess2

        peak2_result = self.peak_potential(peak_potential_child_options(seg2_options))
        E_peak2 = peak2_result["Ep"]
        idx2 = peak2_result["index"]
        x2, y2 = self.analysis_segment_data(seg2_options)
        y_E_peak2 = y2[idx2]

        ΔE = round_sigfigs(abs(E_peak1 - E_peak2), options["sig figs"])
        E_half = midpoint_potential(E_peak1, E_peak2, options["sig figs"])

        peak_data1 = {
            "segment": seg1,
            "Ep": E_peak1,
            "Ep y": y_E_peak1,
        }
        peak_data2 = {
            "segment": seg2,
            "Ep": E_peak2,
            "Ep y": y_E_peak2,
        }

        if options.get("plot", True):
            offset = options.get('offset', 0)
            if not options.get('internal call') and options.get("plot cv", True):
                self._plot_from_analysis_options(options)
                options["new plot"] = False
            x_scale, y_scale = self.xy_scale(options)

            plt.vlines(
                E_half * x_scale,
                y_E_peak1 * y_scale + offset,
                y_E_peak2 * y_scale + offset,
                color='tab:blue',
                linestyle='--'
            )
            plt.scatter(
                [E_peak1 * x_scale, E_peak2 * x_scale],
                np.array([y_E_peak1, y_E_peak2]) * y_scale + offset,
                color='tab:blue',
                zorder=3
            )

        values = {
            "E(1/2)": E_half,
            "ΔE": ΔE,
            "delta E": ΔE,
            "peak 1": peak_data1,
            "peak 2": peak_data2,
        }
        result = _cv_analysis_result(
            self,
            "half_wave_potential",
            values,
            [
                {"metric": "E(1/2)", "value": E_half, "kind": "potential"},
                {"metric": "ΔE", "value": ΔE, "kind": "potential"},
                {"metric": "Ep", "segment": seg1, "value": E_peak1, "kind": "potential"},
                {"metric": "Ep", "segment": seg2, "value": E_peak2, "kind": "potential"},
            ],
            "E(1/2)",
            options,
            diagnostics={"peak 1": peak_data1, "peak 2": peak_data2},
        )
        if options.get("print", True):
            result.show(options)
        return result

    def wave_info(self, options={}):
        """
        Summarize a reversible CV wave using both half-wave and peak analyses.

        Uses:
            - half_wave_potential() for E(1/2) and ΔE
            - peak_info() for Ep, ip, Ep/2, and Δ(Ep - Ep/2)

        Parameters
        ----------
        options : dict, optional
            Analysis/plotting options.

        Returns
        -------
        CVAnalysisResult
            Dictionary-compatible result containing wave-level and peak-level metrics plus a tidy display table.
        """
        options = self._cv_analysis_options(options)

        do_print = options.get("print", True)
        do_plot = options.get("plot", True)
        do_plot_all = options.get("plot all", False)
        do_print_all = options.get("print all", False)
        is_internal = options.get("internal call", False)

        # Plot the CV once at the top level.
        # Child calls then add annotations without re-plotting the trace.
        if do_plot and not is_internal and options.get("plot cv", True):
            self._plot_from_analysis_options(options)
            options["new plot"] = False

        # Use half_wave_potential for E1/2 and ΔE only.
        half_wave_options = options.copy()
        half_wave_options["print"] = do_print_all
        half_wave_options["plot"] = do_plot
        half_wave_options["internal call"] = True
        half_wave_options["new plot"] = False

        half_wave_result = self.half_wave_potential(half_wave_options)
        E_half = half_wave_result["E(1/2)"]
        ΔE = half_wave_result["ΔE"]
        hw_peak1 = half_wave_result["peak 1"]
        hw_peak2 = half_wave_result["peak 2"]

        seg1 = hw_peak1.get("segment", options.get("segment", 1) or 1)
        seg2 = hw_peak2.get("segment", seg1 + 1)

        # Resolve optional separate guesses for peak_info.
        guess = options.get("guess potential", None)
        if isinstance(guess, (tuple, list)) and len(guess) == 2:
            guess1, guess2 = guess
        else:
            guess1 = guess
            guess2 = hw_peak1.get("Ep", guess)

        # Peak 1 full info.
        peak1_options = options.copy()
        peak1_options.pop("segments", None)
        peak1_options["segment"] = seg1
        peak1_options["print"] = do_print_all
        peak1_options["plot"] = do_plot
        peak1_options["internal call"] = True
        peak1_options["new plot"] = False
        if guess1 is not None:
            peak1_options["guess potential"] = guess1

        peak_info1 = self.peak_info(peak1_options)

        # Peak 2 full info.
        peak2_options = options.copy()
        peak2_options.pop("segments", None)
        peak2_options["segment"] = seg2
        peak2_options["print"] = do_print_all
        peak2_options["plot"] = do_plot
        peak2_options["internal call"] = True
        peak2_options["new plot"] = False
        if guess2 is not None:
            peak2_options["guess potential"] = guess2

        peak_info2 = self.peak_info(peak2_options)

        wave_info = {
            "E(1/2)": E_half,
            "ΔE": ΔE,

            "P1 segment": seg1,
            "P1 Ep": peak_info1["Ep"],
            "P1 ip": peak_info1["ip"],
            "P1 Ep/2": peak_info1["Ep/2"],
            "P1 Δ(Ep - Ep/2)": peak_info1["Δ(Ep - Ep/2)"],

            "P2 segment": seg2,
            "P2 Ep": peak_info2["Ep"],
            "P2 ip": peak_info2["ip"],
            "P2 Ep/2": peak_info2["Ep/2"],
            "P2 Δ(Ep - Ep/2)": peak_info2["Δ(Ep - Ep/2)"],
        }

        result = _cv_analysis_result(
            self,
            "wave_info",
            wave_info,
            [
                {"metric": "E(1/2)", "value": E_half, "kind": "potential"},
                {"metric": "ΔE", "value": ΔE, "kind": "potential"},
                {"metric": "Ep", "segment": seg1, "value": peak_info1["Ep"], "kind": "potential"},
                {"metric": "ip", "segment": seg1, "value": peak_info1["ip"], "kind": "current"},
                {"metric": "Ep/2", "segment": seg1, "value": peak_info1["Ep/2"], "kind": "potential"},
                {
                    "metric": "Δ(Ep - Ep/2)",
                    "segment": seg1,
                    "value": peak_info1["Δ(Ep - Ep/2)"],
                    "kind": "potential",
                },
                {"metric": "Ep", "segment": seg2, "value": peak_info2["Ep"], "kind": "potential"},
                {"metric": "ip", "segment": seg2, "value": peak_info2["ip"], "kind": "current"},
                {"metric": "Ep/2", "segment": seg2, "value": peak_info2["Ep/2"], "kind": "potential"},
                {
                    "metric": "Δ(Ep - Ep/2)",
                    "segment": seg2,
                    "value": peak_info2["Δ(Ep - Ep/2)"],
                    "kind": "potential",
                },
            ],
            "E(1/2)",
            options,
            diagnostics={"peak 1": peak_info1, "peak 2": peak_info2},
        )
        if do_print:
            result.show(options)
        return result

    # def wave_infof(self, options={}):
    #     options['normalize'] = False
    #     options = self._cv_analysis_options(options)
    #
    #     # save print and plot options
    #     print_opt = options["print"]
    #     printall_opt = options["print all"]
    #     plot_opt = options["plot"]
    #     plotall_opt = options["plot all"]
    #     options['plot'] = False
    #     options['print'] = False
    #     options["plot all"] = False
    #
    #     # find un-normalized half wave potential
    #     E_half = self.half_wave_potential(options)[0]
    #
    #     # reset print and plot options
    #     options["print"] = print_opt
    #     options["plot"] = plot_opt
    #     options["print all"] = printall_opt
    #     options["plot all"] = plotall_opt
    #
    #     options['normalize'] = True
    #     options['normalize params']["E"] = E_half
    #     options["range"] = [x * 25 for x in options["range"]]
    #
    #     if options["plot"]:
    #         self._plot_from_analysis_options(options)
    #
    #     return self.wave_info(0,options)

    def stats(self):
        """Return CV metadata and scan statistics.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        dict
            CV statistics and metadata.
        
        Examples
        --------
        >>> cv_obj.stats()
        """
        return {
            'solvent': self.solvent,
            'gas': self.gas,
            'compounds': self.compounds,
            'concentrations': self.concentrations,
            'scan rate': self.scan_rate,
            'init E': self.init_E,
            'high E': self.max_E,
            'low E': self.min_E,
            'segments': self.segments,
        }
    
    def txt_stats(self, options=None):
        if options is None:
            options = {}

        stats = super().txt_stats(options)

        if stats.get("gas") is None:
            stats["gas"] = ""

        if stats.get("segments") is not None:
            stats["segments"] = int(stats["segments"])

        if stats.get("scan rate") is not None:
            scaled, label = scale_value(
                stats["scan rate"],
                "V/s",
                selected_unit=options.get("scan rate unit", "auto"),
            )
            stats["scan rate"] = f"{scaled:g} {label}"

        low_E = stats.pop("low E", None)
        high_E = stats.pop("high E", None)
        stats.pop("init E", None)

        if low_E is not None and high_E is not None:
            potential_rounding = options.get("potential rounding", 0.01)
            if potential_rounding not in (None, False):
                potential_rounding = float(potential_rounding)
                if potential_rounding <= 0:
                    raise ValueError("'potential rounding' must be positive, None, or False.")
                low_E = round(float(low_E) / potential_rounding) * potential_rounding
                high_E = round(float(high_E) / potential_rounding) * potential_rounding
            else:
                sig_figs = options.get("sig figs", 4)
                low_E = round_sigfigs(float(low_E), sig_figs)
                high_E = round_sigfigs(float(high_E), sig_figs)
            low_E = f"{low_E:g}"
            high_E = f"{high_E:g}"
            stats["scan window"] = f"[{low_E}, {high_E}]"
        else:
            stats["scan window"] = ""

        return stats

class dpv(echem):
    """Differential pulse voltammetry object with DPV-specific peak analysis.
    
    Parameters
    ----------
    filepath : str or path-like, optional
        DPV text file to parse.
    options : dict or ImportOptions, optional
        Import and parser options. See ``e.describe_options("get_data")``.
    
    Examples
    --------
    >>> dpv_obj = e.dpv(path, {"software": "CH"})
    """
    def __init__(self, filepath=None, options={}):
        super().__init__(filepath, options)
        self.type = "Differential Pulse Voltammetry"

        self.init_E = None
        self.final_E = None
        self.incr_E = None
        self.amplitude = None
        self.pulse_width = None
        self.sample_width = None
        self.pulse_period = None
        self.quiet_time = None
        self.sensitivity = None
        self.comp_R = None
        self.min_E = None
        self.max_E = None

        if filepath is not None:
            self.get_data_from_file(filepath, options)

    def get_data_from_file(self, filepath, options):
        if self.software == "CH":
            self._parse_ch_dpv_file(filepath, options)
        else:
            raise ValueError(f"Unsupported software for DPV: {self.software}")

    def _parse_ch_dpv_file(self, filepath, options):
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            lines = [line.strip() for line in f.readlines()]

        def get_assignment_value(label):
            pattern = rf"^{re.escape(label)}\s*=\s*([\d.eE+\-]+)"
            for line in lines:
                match = re.search(pattern, line)
                if match:
                    return float(match.group(1))
            return None

        self.init_E = get_assignment_value("Init E (V)")
        self.final_E = get_assignment_value("Final E (V)")
        self.incr_E = get_assignment_value("Incr E (V)")
        self.amplitude = get_assignment_value("Amplitude (V)")
        self.pulse_width = get_assignment_value("Pulse Width (sec)")
        self.sample_width = get_assignment_value("Sample Width (sec)")
        self.pulse_period = get_assignment_value("Pulse Period (sec)")
        self.quiet_time = get_assignment_value("Quiet Time (sec)")
        self.sensitivity = get_assignment_value("Sensitivity (A/V)")
        self._parse_ir_compensation_from_lines(lines)
        self.comp_R = get_assignment_value("Comp R (ohm)")

        if "Potential" in self.data.columns and not self.data.empty:
            self.min_E = self.data["Potential"].min()
            self.max_E = self.data["Potential"].max()
            if len(self.data) >= 2:
                self.delta_x = round_sigfigs(
                    abs(self.data["Potential"].iloc[1] - self.data["Potential"].iloc[0]),
                    3,
                )

    def stats(self):
        return {
            'solvent': getattr(self, 'solvent', None),
            'gas': getattr(self, 'gas', None),
            'compounds': getattr(self, 'compounds', []),
            'concentrations': getattr(self, 'concentrations', []),
            'start_x': self.x().iloc[0] if not self.data.empty else None,
            'end_x': self.x().iloc[-1] if not self.data.empty else None,
            'min_x': self.x().min() if not self.data.empty else None,
            'max_x': self.x().max() if not self.data.empty else None,
            'delta_x': self.delta_x,
            'segments': self.segments,
            'init E': getattr(self, 'init_E', None),
            'final E': getattr(self, 'final_E', None),
            'increment E': getattr(self, 'incr_E', None),
            'amplitude': getattr(self, 'amplitude', None),
            'pulse width': getattr(self, 'pulse_width', None),
            'sample width': getattr(self, 'sample_width', None),
            'pulse period': getattr(self, 'pulse_period', None),
            'quiet time': getattr(self, 'quiet_time', None),
            'sensitivity': getattr(self, 'sensitivity', None),
            'comp R': getattr(self, 'comp_R', None),
        }

    def txt_stats(self, options=None):
        if options is None:
            options = {}

        stats = super().txt_stats(options)

        if stats.get("gas") is None:
            stats["gas"] = ""

        low_E = stats.pop("min_x", None)
        high_E = stats.pop("max_x", None)

        for key in [
            "start_x",
            "end_x",
            "delta_x",
            "init E",
            "final E",
            "increment E",
            "quiet time",
            "sensitivity",
            "comp R",
        ]:
            stats.pop(key, None)

        def format_pulse_value(key, value):
            if value in ("", None):
                return ""
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return value
            if not np.isfinite(numeric_value):
                return ""
            if key == "amplitude":
                scaled, unit = scale_value(
                    numeric_value,
                    "V",
                    selected_unit="auto",
                    candidates=("m", "μ", "n", "p"),
                )
            else:
                scaled, unit = scale_value(
                    numeric_value,
                    "s",
                    selected_unit="auto",
                    candidates=("m", "μ", "n", "p"),
                )
            scaled = round_sigfigs(scaled, options.get("sig figs", 4))
            return f"{scaled:g} {unit}"

        for key in ["amplitude", "pulse width", "sample width", "pulse period"]:
            if key in stats:
                stats[key] = format_pulse_value(key, stats[key])

        if low_E is not None and high_E is not None:
            potential_rounding = options.get("potential rounding", 0.01)
            if potential_rounding not in (None, False):
                potential_rounding = float(potential_rounding)
                if potential_rounding <= 0:
                    raise ValueError("'potential rounding' must be positive, None, or False.")
                low_E = round(float(low_E) / potential_rounding) * potential_rounding
                high_E = round(float(high_E) / potential_rounding) * potential_rounding
            else:
                sig_figs = options.get("sig figs", 4)
                low_E = round_sigfigs(float(low_E), sig_figs)
                high_E = round_sigfigs(float(high_E), sig_figs)
            low_E = f"{low_E:g}"
            high_E = f"{high_E:g}"
            stats["scan window"] = f"[{low_E}, {high_E}]"
        else:
            stats["scan window"] = ""

        return stats

    def peak_potential(self, options={}):
        """Find a DPV peak potential near a requested guess.
        
        Parameters
        ----------
        options : dict or PeakPotentialOptions, optional
            Peak-selection and smoothing options. See ``e.describe_options("dpv.peak_potential")``.
        
        Returns
        -------
        tuple
            Peak potential and row index.
        
        Examples
        --------
        >>> E_peak, idx = dpv_obj.peak_potential({"guess potential": -1.5})
        """
        typed_options = PeakPotentialOptions.from_options(options)
        options = typed_options.to_legacy_dict()

        x = self.x(options).to_numpy(dtype=float)
        y = self.y(options).to_numpy(dtype=float)

        exact_potential = options.get("exact potential")
        guess_potential = options.get("guess potential")

        if exact_potential is not None:
            peak_index = int(np.argmin(np.abs(x - float(exact_potential))))
        else:
            extrema, smoothed_y, prom_map, ext_meta = _find_extrema_indices(
                y,
                options,
            )

            if len(extrema) == 0:
                raise ValueError(
                    "peak_potential could not locate any extrema in the DPV trace. "
                    "Check 'guess potential', 'peak prominence', or smoothing settings."
                )

            if guess_potential is not None:
                peak_index = int(extrema[np.argmin(np.abs(x[extrema] - guess_potential))])
            else:
                peak_index = max(extrema, key=lambda idx: prom_map.get(int(idx), 0.0))

        peak_potential = round_sigfigs(x[peak_index], options["sig figs"])

        if options["print"]:
            x_name = self.x(options).name
            x_unit = self.units.get(x_name, "")
            print(f"Ep: {peak_potential} {x_unit}".strip())

        if options["plot"]:
            if not options.get("internal call"):
                self.plot(_plot_options_from_mapping(options))

            x_scale, y_scale = self.xy_scale(options)
            plt.scatter(
                x[peak_index] * x_scale,
                y[peak_index] * y_scale + options.get("offset", 0),
                color="tab:blue",
                zorder=3,
            )

            if options.get("troubleshoot") and exact_potential is None:
                plt.scatter(
                    x[extrema] * x_scale,
                    y[extrema] * y_scale + options.get("offset", 0),
                    color="tab:blue",
                    s=10,
                    zorder=3,
                )
                print(
                    f"SG window={ext_meta['sg window']}, "
                    f"polyorder={ext_meta['sg polyorder']}, "
                    f"prominence={ext_meta['prominence']}"
                )

        return peak_potential, peak_index

    @staticmethod
    def _double_peak_model(E, b0, b1, A1, E1, sigma1, A2, E2, sigma2):
        return (
            b0
            + b1 * E
            + A1 * np.exp(-((E - E1) ** 2) / (2 * sigma1 ** 2))
            + A2 * np.exp(-((E - E2) ** 2) / (2 * sigma2 ** 2))
        )

    def _initial_two_peak_guesses(self, x, y, options):
        guess_potentials = options.get("guess potentials")
        if guess_potentials is None:
            guess_potentials = options.get("guess potential")

        if guess_potentials is not None and not isinstance(guess_potentials, (list, tuple, np.ndarray)):
            guess_potentials = [float(guess_potentials)]

        if guess_potentials is None or len(guess_potentials) < 2:
            dominant, _ = self.peak_potential({
                **options,
                "plot": False,
                "print": False,
            })
            extrema, _smoothed_y, prom_map, _ext_meta = _find_extrema_indices(y, options)
            extrema = [
                int(idx) for idx in extrema
                if abs(float(x[int(idx)]) - float(dominant)) > 3 * float(self.delta_x or 0)
            ]
            if len(extrema) == 0:
                raise ValueError(
                    "fit_overlapping_peaks needs two peak guesses or two resolvable extrema."
                )
            second_idx = max(extrema, key=lambda idx: prom_map.get(int(idx), 0.0))
            guess_potentials = [float(dominant), float(x[second_idx])]

        if len(guess_potentials) != 2:
            raise ValueError("'guess potentials' must contain exactly two potentials.")

        return [float(guess_potentials[0]), float(guess_potentials[1])]

    def _resolve_two_peak_guesses_for_active_axis(
        self,
        guess_potentials,
        x_min,
        x_max,
        axis_tol,
    ):
        def within_axis(values, lower, upper):
            return all(lower - axis_tol <= float(value) <= upper + axis_tol for value in values)

        if within_axis(guess_potentials, x_min, x_max):
            return [float(guess) for guess in guess_potentials]

        if self.has_reference_shift() and "Potential" in self.data.columns:
            raw_x = self.data["Potential"].to_numpy(dtype=float)
            raw_x = raw_x[np.isfinite(raw_x)]
            if len(raw_x) > 0:
                raw_min = float(np.nanmin(raw_x))
                raw_max = float(np.nanmax(raw_x))
                shifted_guesses = [
                    float(guess) - float(self.reference_shift)
                    for guess in guess_potentials
                ]
                if within_axis(guess_potentials, raw_min, raw_max) and within_axis(
                    shifted_guesses,
                    x_min,
                    x_max,
                ):
                    return shifted_guesses

        outside = [
            float(guess)
            for guess in guess_potentials
            if guess < x_min - axis_tol or guess > x_max + axis_tol
        ]
        raise ValueError(
            "'guess potentials' contains values outside the active x-axis range "
            f"[{x_min:g}, {x_max:g}]: {outside}. Use potentials in the same "
            "axis returned by x(), or set the x-axis option to match your guesses."
        )

    def _resolve_fit_window_for_active_axis(self, fit_window, x_min, x_max, axis_tol):
        if fit_window is None:
            return None
        if not isinstance(fit_window, (list, tuple, np.ndarray)) or len(fit_window) != 2:
            raise ValueError("'fit window' must contain exactly two potentials.")

        values = [float(fit_window[0]), float(fit_window[1])]
        lower = min(values)
        upper = max(values)
        if x_min - axis_tol <= lower <= x_max + axis_tol and x_min - axis_tol <= upper <= x_max + axis_tol:
            return [lower, upper]

        if self.has_reference_shift() and "Potential" in self.data.columns:
            raw_x = self.data["Potential"].to_numpy(dtype=float)
            raw_x = raw_x[np.isfinite(raw_x)]
            if len(raw_x) > 0:
                raw_min = float(np.nanmin(raw_x))
                raw_max = float(np.nanmax(raw_x))
                if raw_min - axis_tol <= lower <= raw_max + axis_tol and raw_min - axis_tol <= upper <= raw_max + axis_tol:
                    shifted = [
                        float(value) - float(self.reference_shift)
                        for value in values
                    ]
                    return [min(shifted), max(shifted)]

        raise ValueError(
            "'fit window' contains values outside the active x-axis range "
            f"[{x_min:g}, {x_max:g}]."
        )

    @staticmethod
    def _center_bounds_from_window(center_window, guesses, x_min, x_max):
        if center_window in (None, False):
            return [(x_min, x_max), (x_min, x_max)]

        if isinstance(center_window, (int, float, np.number)):
            widths = [float(center_window), float(center_window)]
        elif isinstance(center_window, (list, tuple, np.ndarray)) and len(center_window) == 2:
            widths = [float(center_window[0]), float(center_window[1])]
        else:
            raise ValueError("'center window' must be a positive number or two positive numbers.")

        if any(not np.isfinite(width) or width <= 0 for width in widths):
            raise ValueError("'center window' must be positive.")

        bounds = []
        for guess, width in zip(guesses, widths):
            lower = max(x_min, float(guess) - width)
            upper = min(x_max, float(guess) + width)
            if lower >= upper:
                raise ValueError("'center window' leaves no feasible center range for a guessed peak.")
            bounds.append((lower, upper))
        return bounds

    def fit_overlapping_peaks(self, options={}):
        options = {} if options is None else dict(options)

        x = self.x(options).to_numpy(dtype=float)
        y = self.y(options).to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(x) < 8:
            raise ValueError("fit_overlapping_peaks needs at least 8 finite data points.")

        guess_potentials = self._initial_two_peak_guesses(x, y, options)

        edge_count = max(3, int(np.ceil(0.12 * len(x))))
        edge_idx = np.r_[0:edge_count, len(x) - edge_count:len(x)]
        b1, b0 = np.polyfit(x[edge_idx], y[edge_idx], 1)
        baseline = b0 + b1 * x
        residual = y - baseline

        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        x_span = abs(x_max - x_min)
        dx_data = np.nanmedian(np.abs(np.diff(np.sort(x))))
        dx = abs(float(self.delta_x or dx_data or x_span / len(x)))
        sigma_min = float(options.get("sigma min", max(dx / 2, x_span / 1000)))
        sigma_max = float(options.get("sigma max", x_span))
        if not np.isfinite(sigma_min) or not np.isfinite(sigma_max) or sigma_max <= sigma_min:
            raise ValueError("'sigma max' must be greater than 'sigma min'.")
        sigma_guess = float(options.get("sigma guess", max(3 * dx, x_span / 30)))
        if not np.isfinite(sigma_guess):
            sigma_guess = max(3 * dx, x_span / 30)
        sigma_guess = float(np.clip(sigma_guess, sigma_min, sigma_max))

        axis_tol = max(dx * 1e-6, x_span * 1e-9, np.finfo(float).eps)
        guess_potentials = self._resolve_two_peak_guesses_for_active_axis(
            guess_potentials,
            x_min,
            x_max,
            axis_tol,
        )

        fit_window = self._resolve_fit_window_for_active_axis(
            options.get("fit window"),
            x_min,
            x_max,
            axis_tol,
        )
        if fit_window is not None:
            fit_min, fit_max = fit_window
            fit_mask = (x >= fit_min - axis_tol) & (x <= fit_max + axis_tol)
            x = x[fit_mask]
            y = y[fit_mask]
            if len(x) < 8:
                raise ValueError("'fit window' leaves fewer than 8 finite data points.")

            x_min = float(np.nanmin(x))
            x_max = float(np.nanmax(x))
            x_span = abs(x_max - x_min)
            dx_data = np.nanmedian(np.abs(np.diff(np.sort(x))))
            dx = abs(float(dx_data or x_span / len(x)))
            axis_tol = max(dx * 1e-6, x_span * 1e-9, np.finfo(float).eps)
            if not all(x_min - axis_tol <= guess <= x_max + axis_tol for guess in guess_potentials):
                raise ValueError("'guess potentials' must fall inside 'fit window'.")

            edge_count = max(3, int(np.ceil(0.12 * len(x))))
            edge_idx = np.r_[0:edge_count, len(x) - edge_count:len(x)]
            b1, b0 = np.polyfit(x[edge_idx], y[edge_idx], 1)
            baseline = b0 + b1 * x
            residual = y - baseline
            sigma_min = float(options.get("sigma min", max(dx / 2, x_span / 1000)))
            sigma_max = float(options.get("sigma max", x_span))
            if not np.isfinite(sigma_min) or not np.isfinite(sigma_max) or sigma_max <= sigma_min:
                raise ValueError("'sigma max' must be greater than 'sigma min'.")
            sigma_guess = float(options.get("sigma guess", max(3 * dx, x_span / 30)))
            if not np.isfinite(sigma_guess):
                sigma_guess = max(3 * dx, x_span / 30)
            sigma_guess = float(np.clip(sigma_guess, sigma_min, sigma_max))

        guess_potentials = [
            float(np.clip(float(guess), x_min, x_max))
            for guess in guess_potentials
        ]
        center_bounds = self._center_bounds_from_window(
            options.get("center window"),
            guess_potentials,
            x_min,
            x_max,
        )

        def _move_inside(value, lower_bound, upper_bound):
            value = float(value)
            if value <= lower_bound:
                return float(np.nextafter(lower_bound, upper_bound))
            if value >= upper_bound:
                return float(np.nextafter(upper_bound, lower_bound))
            return value

        peak_params = []
        for guess in guess_potentials:
            idx = int(np.argmin(np.abs(x - guess)))
            amp_guess = float(residual[idx])
            if amp_guess == 0:
                amp_guess = float(np.nanmin(residual))
            peak_params.extend([
                amp_guess,
                _move_inside(guess, x_min, x_max),
                _move_inside(sigma_guess, sigma_min, sigma_max),
            ])

        p0 = [float(b0), float(b1), *peak_params]
        lower = [
            -np.inf, -np.inf,
            -np.inf, center_bounds[0][0], sigma_min,
            -np.inf, center_bounds[1][0], sigma_min,
        ]
        upper = [
            np.inf, np.inf,
            np.inf, center_bounds[0][1], sigma_max,
            np.inf, center_bounds[1][1], sigma_max,
        ]

        try:
            popt, pcov = curve_fit(
                self._double_peak_model,
                x,
                y,
                p0=p0,
                bounds=(lower, upper),
                maxfev=int(options.get("maxfev", 20000)),
            )
        except ValueError as exc:
            if "infeasible" in str(exc) or "outside" in str(exc):
                raise ValueError(
                    "fit_overlapping_peaks initial guesses are infeasible for the "
                    "active x-axis or sigma bounds. Check 'guess potentials', "
                    "'sigma guess', 'sigma min', and 'sigma max'."
                ) from exc
            raise

        b0_fit, b1_fit, A1, E1, sigma1, A2, E2, sigma2 = popt
        rows = [
            {
                "component": "peak 1",
                "potential (V)": E1,
                "amplitude (A)": A1,
                "sigma (V)": abs(sigma1),
                "baseline current (A)": b0_fit + b1_fit * E1,
                "current (A)": self._double_peak_model(E1, *popt),
            },
            {
                "component": "peak 2",
                "potential (V)": E2,
                "amplitude (A)": A2,
                "sigma (V)": abs(sigma2),
                "baseline current (A)": b0_fit + b1_fit * E2,
                "current (A)": self._double_peak_model(E2, *popt),
            },
        ]
        result = pd.DataFrame(rows).sort_values("potential (V)", ascending=False).reset_index(drop=True)
        result["component"] = [f"peak {i}" for i in range(1, len(result) + 1)]
        result.attrs["baseline"] = {"intercept": float(b0_fit), "slope": float(b1_fit)}
        result.attrs["fit parameters"] = [float(value) for value in popt]
        result.attrs["covariance"] = pcov.tolist()

        if options.get("print", True):
            print(result.to_string(index=False))

        if options.get("plot", True):
            legend_requested = options.get("legend", True)
            plot_options = _plot_options_from_mapping(options)
            plot_options["legend"] = False
            data_label = options.get("data label", "DPV Data")
            plot_options["label"] = "_nolegend_" if data_label in (None, False) else str(data_label)
            self.plot(plot_options)
            x_fit = np.linspace(x_min, x_max, int(options.get("fit points", 500)))
            baseline_fit = b0_fit + b1_fit * x_fit
            peak1_fit = A1 * np.exp(-((x_fit - E1) ** 2) / (2 * sigma1 ** 2))
            peak2_fit = A2 * np.exp(-((x_fit - E2) ** 2) / (2 * sigma2 ** 2))
            y_fit = self._double_peak_model(x_fit, *popt)
            x_scale, y_scale = self.xy_scale(options)
            component_colors = options.get("component colors", ["tab:orange", "tab:green"])
            if isinstance(component_colors, str):
                component_colors = [component_colors, component_colors]
            while len(component_colors) < 2:
                component_colors.append(component_colors[-1] if component_colors else "tab:orange")
            plt.plot(
                x_fit * x_scale,
                y_fit * y_scale + options.get("offset", 0),
                color=_first_fit_color(options),
                linestyle=options.get("fit linestyle", "--"),
                label=options.get("fit label", "Total Fit") if legend_requested else None,
            )
            plt.plot(
                x_fit * x_scale,
                (baseline_fit + peak1_fit) * y_scale + options.get("offset", 0),
                color=component_colors[0],
                linestyle=options.get("component linestyle", ":"),
                label=options.get("peak 1 label", "Peak 1") if legend_requested else None,
            )
            plt.plot(
                x_fit * x_scale,
                (baseline_fit + peak2_fit) * y_scale + options.get("offset", 0),
                color=component_colors[1],
                linestyle=options.get("component linestyle", ":"),
                label=options.get("peak 2 label", "Peak 2") if legend_requested else None,
            )
            plt.scatter(
                result["potential (V)"].to_numpy() * x_scale,
                result["current (A)"].to_numpy() * y_scale + options.get("offset", 0),
                color=options.get("peak color", "tab:red"),
                zorder=3,
                label="_nolegend_",
            )
            if legend_requested:
                plt.legend(fontsize=options.get("legend fontsize") or _default_legend_fontsize())

        return result

    closely_spaced_peaks = fit_overlapping_peaks

DPV = dpv

class cp(echem):
    """Chronopotentiometry object with cycle-aware plotting helpers.
    
    Parameters
    ----------
    filepath : str or path-like, optional
        CP text file to parse.
    options : dict or ImportOptions, optional
        Import and parser options. See ``e.describe_options("get_data")``.
    
    Examples
    --------
    >>> cp_obj = e.cp(path, {"software": "CH"})
    """
    def __init__(self, filepath=None, options={}):
        super().__init__(filepath, options)

        self.type = "Chronopotentiometry"

        # Initialize CP-specific variables
        self.cathodic_current = None
        self.anodic_current = None
        self.init_PN = None
        self.high_E_limit = None
        self.low_E_limit = None
        self.cathodic_time = None
        self.anodic_time = None
        self.segments = None
        self.sample_int = None
        self.quiet_time = None

        # General electrochemical variables
        self.init_E = None
        self.final_E = None
        self.min_E = None
        self.max_E = None

        # Parse data depending on software
        self.get_data_from_file(filepath, options)

    def get_data_from_file(self, filepath, options):
        if self.software == "CH":
            self._parse_ch_cp_file(filepath, options)
        elif str(self.software).lower().replace("-", "") == "eclab":
            self._parse_eclab_cp_file(filepath, options)
        elif self.software == "BASI":
            self._parse_basi_cp_file(filepath, options)
        else:
            raise ValueError(f"Unsupported software for CP: {self.software}")

    def _parse_ch_cp_file(self, filepath, options):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header_lines = []
        data_start = 0
        for i, line in enumerate(lines):
            if "Time" in line and "Potential" in line:
                data_start = i
                break
            header_lines.append(line.strip())

        for line in header_lines:
            if m := re.search(r'Cathodic Current \(A\) = ([\d.eE+-]+)', line):
                self.cathodic_current = float(m.group(1))
            elif m := re.search(r'Anodic Current \(A\) = ([\d.eE+-]+)', line):
                self.anodic_current = float(m.group(1))
            elif m := re.search(r'Init P/N = (\w+)', line):
                self.init_PN = m.group(1)
            elif m := re.search(r'Data Storage Interval \(s\) = ([\d.eE+-]+)', line):
                self.sample_int = float(m.group(1))
            elif m := re.search(r'High E Limit \(V\) = ([\d.eE+-]+)', line):
                self.high_E_limit = float(m.group(1))
            elif m := re.search(r'Low E Limit \(V\) = ([\d.eE+-]+)', line):
                self.low_E_limit = float(m.group(1))
            elif m := re.search(r'Cathodic Time \(s\) = ([\d.eE+-]+)', line):
                self.cathodic_time = float(m.group(1))
            elif m := re.search(r'Anodic Time \(s\) = ([\d.eE+-]+)', line):
                self.anodic_time = float(m.group(1))
            elif m := re.search(r'Segment = ([\d]+)', line):
                self.segments = int(m.group(1))
        self.quiet_time = _parse_quiet_time_from_lines(header_lines)

        # Load data into DataFrame
        df = pd.read_csv(filepath, skiprows=data_start + 1, names=["Time", "Potential"])

        # Patch any repeated time entries
        times = df["Time"].values.copy()
        for i in range(1, len(times)):
            # if the stamp didn’t advance (or went backwards), nudge it forward
            if times[i] <= times[i - 1]:
                times[i] = times[i - 1] + self.sample_int
        df["Time"] = times

        # Store back and set units
        self.data = df
        self.units["Time"] = "s"
        self.units["Potential"] = "V"

        # Estimate init/final/min/max potentials and delta_x
        self.init_E = self.data["Potential"].iloc[0]
        self.final_E = self.data["Potential"].iloc[-1]
        self.min_E = self.data["Potential"].min()
        self.max_E = self.data["Potential"].max()
        if len(self.data) >= 2:
            self.delta_x = self.data["Time"].iloc[1] - self.data["Time"].iloc[0]

    def _parse_eclab_cp_file(self, filepath, options):
        with open(filepath, "r", encoding="ISO-8859-1") as f:
            lines = f.readlines()

        self.type = "Chronopotentiometry"

        skiprows = None
        for line in lines:
            if "Nb header lines" in line:
                try:
                    skiprows = int(line.split(":", 1)[1].strip().split()[0])
                    break
                except Exception as exc:
                    raise ValueError(
                        "Could not parse number of header lines in EC-Lab CP file."
                    ) from exc

        if skiprows is None:
            raise ValueError("Header line count not found in EC-Lab CP file.")

        header_lines = lines[: max(skiprows - 1, 0)]
        self.quiet_time = _parse_quiet_time_from_lines(header_lines)

        def _parse_step_values(label):
            for idx, line in enumerate(header_lines):
                if line.strip().startswith(label):
                    values = []
                    for token in re.split(r"\s+", line.strip())[1:]:
                        try:
                            values.append(float(token))
                        except ValueError:
                            pass

                    units = []
                    if idx + 1 < len(header_lines):
                        next_line = header_lines[idx + 1].strip()
                        if next_line.lower().startswith(f"unit {label.lower()}"):
                            units = re.split(r"\s+", next_line)[2:]

                    return values, units

            return [], []

        def _convert_by_unit(value, unit):
            unit = "" if unit is None else str(unit).strip()
            try:
                return value * get_conversion_factor(unit)
            except Exception:
                return value

        is_values, is_units = _parse_step_values("Is")
        converted_currents = [
            _convert_by_unit(value, is_units[idx] if idx < len(is_units) else "")
            for idx, value in enumerate(is_values)
        ]
        nonzero_currents = [value for value in converted_currents if value != 0]
        if nonzero_currents:
            positive = [value for value in nonzero_currents if value > 0]
            negative = [value for value in nonzero_currents if value < 0]
            if positive:
                self.anodic_current = positive[0]
            if negative:
                self.cathodic_current = negative[0]

        df_raw = pd.read_csv(filepath, sep="\t", skiprows=skiprows - 1, encoding="latin1")
        df_raw = df_raw.dropna(how="all", axis=1).dropna(how="all").reset_index(drop=True)
        normalized_cols = {str(col).strip().lower(): col for col in df_raw.columns}

        time_col_key = next(
            (key for key in normalized_cols if key.startswith("time") and "/" in key),
            None,
        )
        potential_col_key = next(
            (key for key in normalized_cols if "ewe" in key and "/" in key),
            None,
        )

        if time_col_key is None or potential_col_key is None:
            raise ValueError(
                "Could not find expected EC-Lab CP time/potential columns. "
                f"Available columns: {list(df_raw.columns)}"
            )

        time_col = normalized_cols[time_col_key]
        potential_col = normalized_cols[potential_col_key]

        time_unit = time_col.split("/")[-1].strip()
        potential_unit = potential_col.split("/")[-1].strip()

        time_numeric = pd.to_numeric(df_raw[time_col], errors="coerce")
        if time_numeric.notna().all():
            time_values = time_numeric * get_conversion_factor(time_unit)
        else:
            parsed_time = pd.to_datetime(df_raw[time_col], errors="coerce")
            if parsed_time.isna().any():
                raise ValueError(
                    f"Could not parse EC-Lab CP time column '{time_col}' as seconds or timestamps."
                )
            time_values = (parsed_time - parsed_time.iloc[0]).dt.total_seconds()

        potential_values = (
            pd.to_numeric(df_raw[potential_col], errors="coerce")
            * get_conversion_factor(potential_unit)
        )

        df = pd.DataFrame({
            "Time": time_values,
            "Potential": potential_values,
        }).dropna().reset_index(drop=True)

        if df.empty:
            raise ValueError("EC-Lab CP file did not contain any numeric time/potential rows.")

        self.data = df
        self.units = {
            key: value
            for key, value in self.units.items()
            if key not in {"Current", "<I>"}
        }
        self.units["Time"] = "s"
        self.units["Potential"] = "V"

        self.init_E = df["Potential"].iloc[0]
        self.final_E = df["Potential"].iloc[-1]
        self.min_E = df["Potential"].min()
        self.max_E = df["Potential"].max()

        if len(df) >= 2:
            diffs = np.diff(df["Time"].to_numpy(dtype=float))
            positive_diffs = diffs[diffs > 0]
            if len(positive_diffs) > 0:
                self.sample_int = float(np.median(positive_diffs))
                self.delta_x = self.sample_int

        if "ns" in normalized_cols:
            ns = pd.to_numeric(df_raw[normalized_cols["ns"]], errors="coerce").dropna()
            if not ns.empty:
                self.segments = int(ns.nunique())

    def _parse_basi_cp_file(self, filepath, options):
        with open(filepath, "r", encoding="ISO-8859-1") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        self.quiet_time = _parse_quiet_time_from_lines(lines)

        data_header_idx = next(
            (
                i + 1
                for i, line in enumerate(lines)
                if line.strip().lower() == "[begin data]" and i + 1 < len(lines)
            ),
            None,
        )
        if data_header_idx is None:
            data_header_idx = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if line.lower().startswith("time") and "potential" in line.lower()
                ),
                None,
            )

        if data_header_idx is not None:
            delimiter = self.infer_delimiter(lines[data_header_idx])
            df = pd.read_csv(
                filepath,
                sep=delimiter,
                skiprows=data_header_idx,
                header=0,
                engine="python",
                encoding="ISO-8859-1",
            )
        else:
            df = pd.read_csv(
                filepath,
                names=["Time", "Potential"],
                engine="python",
                encoding="ISO-8859-1",
            )

        df = df.dropna(how="all", axis=1).dropna(how="all").reset_index(drop=True)

        updated_columns = {}
        parsed_units = {}
        for col in df.columns:
            col_str = str(col).strip()
            if "/" in col_str:
                name, unit = map(str.strip, col_str.split("/", 1))
            else:
                name, unit = col_str, None

            name_lower = name.lower()
            if name_lower.startswith("time"):
                clean_name = "Time"
                clean_unit = "s" if unit in (None, "sec") else unit
            elif "potential" in name_lower or name_lower.startswith("ewe"):
                clean_name = "Potential"
                clean_unit = "V" if unit is None else unit
            else:
                clean_name = name.strip()
                clean_unit = unit

            updated_columns[col] = clean_name
            if clean_unit is not None:
                parsed_units[clean_name] = clean_unit
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.rename(columns=updated_columns)
        if not {"Time", "Potential"}.issubset(df.columns):
            raise ValueError(
                "Could not find expected BASI CP time/potential columns. "
                f"Available columns: {list(df.columns)}"
            )

        df = df[["Time", "Potential"]].dropna().reset_index(drop=True)
        if df.empty:
            raise ValueError("BASI CP file did not contain any numeric time/potential rows.")

        time_unit = parsed_units.get("Time", "s")
        potential_unit = parsed_units.get("Potential", "V")
        df["Time"] = df["Time"] * get_conversion_factor(time_unit)
        df["Potential"] = df["Potential"] * get_conversion_factor(potential_unit)

        self.data = df
        self.units["Time"] = "s"
        self.units["Potential"] = "V"

        self.init_E = df["Potential"].iloc[0]
        self.final_E = df["Potential"].iloc[-1]
        self.min_E = df["Potential"].min()
        self.max_E = df["Potential"].max()
        if len(df) >= 2:
            diffs = np.diff(df["Time"].to_numpy(dtype=float))
            positive_diffs = diffs[diffs > 0]
            if len(positive_diffs) > 0:
                self.sample_int = float(np.median(positive_diffs))
                self.delta_x = self.sample_int

    def stats(self):
        """
        Return a dictionary of key chronopotentiometry statistics.
        """
        # total experiment duration, if Time column exists
        try:
            total_dur = float(self.data["Time"].iloc[-1] - self.data["Time"].iloc[0])
        except Exception:
            total_dur = None

        return {
            "solvent": getattr(self, "solvent", None),
            "gas": getattr(self, "gas", None),
            "compounds": getattr(self, "compounds", None),
            "concentrations": getattr(self, "concentrations", None),
            "Cathodic current (A)": getattr(self, "cathodic_current", None),
            "Anodic current (A)": getattr(self, "anodic_current", None),
            "Initial P/N": getattr(self, "init_PN", None),
            "High E limit (V)": getattr(self, "high_E_limit", None),
            "Low E limit (V)": getattr(self, "low_E_limit", None),
            "Cathodic time (s)": getattr(self, "cathodic_time", None),
            "Anodic time (s)": getattr(self, "anodic_time", None),
            "Segments": getattr(self, "segments", None),
            "Sample interval (s)": getattr(self, "sample_int", None),
            "Quiet time (s)": getattr(self, "quiet_time", None),
            "Initial potential (V)": getattr(self, "init_E", None),
            "Final potential (V)": getattr(self, "final_E", None),
            "Min potential (V)": getattr(self, "min_E", None),
            "Max potential (V)": getattr(self, "max_E", None),
            "Total duration (s)": total_dur,
        }

    def txt_stats(self, options=None):
        if options is None:
            options = {}

        stats = {
            "exp type": _exp_type_short(getattr(self, "type", "")),
        }
        for key in [
            "Cathodic current (A)",
            "Anodic current (A)",
            "Initial P/N",
            "High E limit (V)",
            "Low E limit (V)",
            "Cathodic time (s)",
            "Anodic time (s)",
            "Segments",
            "Sample interval (s)",
            "Quiet time (s)",
            "Initial potential (V)",
            "Final potential (V)",
            "Min potential (V)",
            "Max potential (V)",
            "Total duration (s)",
        ]:
            stats.pop(key, None)

        stats["technique"] = "CP"
        stats["solvent"] = getattr(self, "solvent", None)
        stats["gas"] = getattr(self, "gas", None) or ""
        stats["compounds"] = self.combine_concs_chems(
            list(getattr(self, "concentrations", []) or []),
            list(getattr(self, "compounds", []) or []),
            options,
        )
        n_cycles = None
        if getattr(self, "segments", None) is not None:
            try:
                n_cycles = int(np.ceil(float(self.segments) / 2))
            except (TypeError, ValueError):
                n_cycles = None
        if n_cycles is not None:
            stats["cycles"] = f"{n_cycles} cycle" if n_cycles == 1 else f"{n_cycles} cycles"

        low = getattr(self, "low_E_limit", None)
        high = getattr(self, "high_E_limit", None)
        if low is not None and high is not None:
            stats["potential limits"] = f"{float(low):g} to {float(high):g} V"

        return stats

    def get_cycles(self, options={}):
        """Split chronopotentiometry data into charge/discharge cycles.
        
        Parameters
        ----------
        options : dict, optional
            Cycle detection and display options. See ``e.describe_options("cp.get_cycles")``.
        
        Returns
        -------
        list
            Per-cycle dataframes or cycle records.
        
        Examples
        --------
        >>> cycles = cp_obj.get_cycles()
        """
        # reconstruct monotonic time array using storage interval
        n = len(self.data)
        t = np.arange(n) * self.sample_int
        v = self.data['Potential'].values

        # compute slope and identify breakpoints
        dt = np.diff(t)
        dv = np.diff(v)
        dt_safe = np.where(dt == 0, np.mean(dt), dt)
        slope = np.nan_to_num(dv / dt_safe)

        # determine how many breaks (half‐steps)
        num_breaks = max(int(self.segments) - 1, 0)
        num_breaks = options.get('slope peaks', num_breaks)
        # pick top-|slope| peaks
        peak_idxs = (np.sort(np.argsort(np.abs(slope))[-num_breaks:])
                     if num_breaks > 0 else [])
        change_pts = peak_idxs + 1
        # include start and end
        idx_changes = np.unique(np.concatenate(([0], change_pts, [len(t)])))
        # segment index lists
        seg_idxs = [np.arange(idx_changes[i], idx_changes[i+1])
                    for i in range(len(idx_changes)-1)]

        return {'t': t, 'v': v, 'seg_idxs': seg_idxs}
        

    def cycle_info(self, options={}):
        """
        Compute per‐cycle summary using get_cycles segmentation.
        Returns a DataFrame with columns:
          Cycle, Duration (s), Discharge Capacity (mA·h), Charge Capacity (mA·h),
          Percent Capacity (%), Coulombic Efficiency (%),
          Discharge Energy (mWh), Charge Energy (mWh),
          Discharge Potential (V), Charge Potential (V)
        """
        data = self.get_cycles(options)
        t = data['t']; v = data['v']; seg_idxs = data['seg_idxs']

        # per‐segment metrics
        durations = []
        Q_dis = []; Q_ch = []
        E_dis = []; E_ch = []
        pot_dis = []; pot_ch = []

        for i, idxs in enumerate(seg_idxs):
            ti = t[idxs]; vi = v[idxs]
            dur = ti[-1] - ti[0]
            durations.append(dur)
            p_end = vi[-1]
            if i % 2 == 0:
                # discharge
                I = self.cathodic_current
                Q = abs(I) * dur / 3600 * 1000
                dt_seg = np.diff(ti)
                energy_J = abs(I * _integrate_trapezoid(vi, ti))
                energy_mWh = energy_J / 3600 * 1000
                Q_dis.append(Q); E_dis.append(energy_mWh); pot_dis.append(p_end)
            else:
                # charge
                I = self.anodic_current
                Q = abs(I) * dur / 3600 * 1000
                dt_seg = np.diff(ti)
                energy_J = abs(I * _integrate_trapezoid(vi, ti))
                energy_mWh = energy_J / 3600 * 1000
                Q_ch.append(Q); E_ch.append(energy_mWh); pot_ch.append(p_end)

        # align cycles by discharge count
        n_dis = len(Q_dis)
        cycles = np.arange(1, n_dis + 1)
        # full‐cycle durations
        durations_cyc = [(durations[2*k] if 2*k < len(durations) else 0)
                         + (durations[2*k+1] if 2*k+1 < len(durations) else 0)
                         for k in range(n_dis)]

        # pad all arrays to same length
        arrays = [
            np.array(Q_dis),
            np.array(Q_ch),
            np.array(E_dis),
            np.array(E_ch),
            np.array(pot_dis),
            np.array(pot_ch)
        ]
        max_len = max(arr.shape[0] for arr in arrays)

        def _pad(arr):
            return np.concatenate([arr, np.full(max_len - arr.shape[0], np.nan)])

        Q_dis_arr, Q_ch_arr, E_dis_arr, E_ch_arr, pot_dis_arr, pot_ch_arr = map(_pad, arrays)

        # percent/absolute capacity
        charge_unit = '(mA·h)'
        if options.get('percent capacity', False) and n_dis > 0:
            charge_unit = '(%)'

            def _normalize_capacity(arr):
                finite_nonzero = arr[np.isfinite(arr) & (arr != 0)]
                if len(finite_nonzero) == 0:
                    return np.full_like(arr, np.nan, dtype=float)
                return arr / finite_nonzero[0] * 100

            Q_dis_arr = _normalize_capacity(Q_dis_arr)
            Q_ch_arr = _normalize_capacity(Q_ch_arr)

        # coulombic & energy efficiency
        CE = Q_dis_arr / Q_ch_arr * 100
        EE = E_dis_arr / E_ch_arr * 100

        # build DataFrame
        df = pd.DataFrame({
            'Cycle': cycles,
            'Duration (s)': durations_cyc,
            f'Discharge Capacity {charge_unit}': Q_dis_arr,
            f'Charge Capacity {charge_unit}': Q_ch_arr,
            'Coulombic Efficiency (%)': CE,
            'Discharge Energy (mWh)': E_dis_arr,
            'Charge Energy (mWh)': E_ch_arr,
            'Energy Efficiency (%)': EE,
            'Discharge Potential (V)': pot_dis_arr,
            'Charge Potential (V)': pot_ch_arr
        })
        return df

    def plot_cycles(self, options={}, **mpl_kwargs):
        """Plot chronopotentiometry cycles.
        
        Parameters
        ----------
        options : dict, optional
            Cycle selection and plotting options. See ``e.describe_options("cp.plot_cycles")``.
        **mpl_kwargs
            Additional keyword arguments passed to Matplotlib.
        
        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the cycle plot.
        
        Examples
        --------
        >>> cp_obj.plot_cycles({"plot": True})
        """
        # Decide whether to make a fresh figure+axes or reuse
        if options.get('new plot', False):
            fig, ax = plt.subplots()
        else:
            ax = plt.gca()
            fig = ax.figure

        data = self.get_cycles(options)
        t_full = data['t']
        v_full = data['v']
        seg_idxs = data['seg_idxs']
        # determine total number of full cycles (discharge segments)
        total_cycles = len(seg_idxs) // 2 + len(seg_idxs) % 2
        sel_cycles = self._resolve_cycles_option(options.get('cycles'), total_cycles)
        
        # choose segment type
        seg_type = options.get('segment', 'full')
        if seg_type not in {'full', 'discharge', 'charge', 'both'}:
            raise ValueError("Invalid 'segment' option")

        x_mode = options.get('x-axis', 'capacity')
        gradient_enabled = str(options.get("color mode", "auto")).strip().lower() == "gradient"
        cycle_colors = {}
        color_spec = {"plot labels": [], "gradient groups": [], "discrete indices": []}
        if gradient_enabled and len(sel_cycles) > 0:
            cmap = _get_group_cmap(0, 1, options)
            values = np.asarray(sel_cycles, dtype=float)
            if len(values) == 1:
                norm = mpl.colors.Normalize(vmin=values[0] - 0.5, vmax=values[0] + 0.5)
            else:
                norm = mpl.colors.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
            for cycle, value in zip(sel_cycles, values):
                cycle_colors[cycle] = mpl.colors.to_hex(cmap(norm(value)))
            cycle_labels = [f"Cycle {int(c)}" if float(c).is_integer() else f"Cycle {c:g}" for c in values]
            tick_mode = str(options.get("colorbar tick labels", "endpoints")).strip().lower()
            ticks = values.copy()
            if tick_mode == "all" or len(cycle_labels) <= 2:
                ticklabels = cycle_labels
            else:
                ticklabels = [""] * len(cycle_labels)
                ticklabels[0] = cycle_labels[0]
                ticklabels[-1] = cycle_labels[-1]
            endpoint_ticks = [ticks[0], ticks[-1]] if len(ticks) > 1 else list(ticks)
            endpoint_ticklabels = (
                [cycle_labels[0], cycle_labels[-1]]
                if len(cycle_labels) > 1
                else list(cycle_labels)
            )
            color_spec = {
                "plot labels": ["_nolegend_"] * len(sel_cycles),
                "gradient groups": [{
                    "indices": list(range(len(sel_cycles))),
                    "values": values,
                    "legend unit": "",
                    "gradient by": "cycle",
                    "legend title": "",
                    "legend context line": "",
                    "cmap": cmap,
                    "norm": norm,
                    "resolved scale": "linear",
                    "ticks": ticks,
                    "ticklabels": ticklabels,
                    "endpoint ticks": endpoint_ticks,
                    "endpoint ticklabels": endpoint_ticklabels,
                }],
                "discrete indices": [],
            }

        def _cycle_plot_kwargs(cycle):
            kwargs = dict(mpl_kwargs)
            if cycle in cycle_colors and "color" not in kwargs:
                kwargs["color"] = cycle_colors[cycle]
            return kwargs

        for cycle in sel_cycles:
            # indices for discharge and charge
            dis_idx = seg_idxs[2 * (cycle - 1)] if 2 * (cycle - 1) < len(seg_idxs) else np.array([])
            chg_idx = seg_idxs[2 * (cycle - 1) + 1] if 2 * (cycle - 1) + 1 < len(seg_idxs) else np.array([])

            if seg_type == 'full':
                # combine both segments
                idxs = np.concatenate((dis_idx, chg_idx)) if chg_idx.size > 0 else dis_idx
                if idxs.size == 0:
                    continue
                t_seg = t_full[idxs] - t_full[idxs][0]
                v_seg = v_full[idxs]
                label = f"Cycle {cycle}"
                # x-axis mapping
                if x_mode == 'capacity':
                    # discharge capacity (mA·h)
                    t_dis = t_full[dis_idx] - t_full[dis_idx][0]
                    Q_dis = abs(self.cathodic_current) * t_dis / 3600 * 1000

                    # charge capacity offset by last discharge Q
                    if chg_idx.size > 0:
                        t_chg = t_full[chg_idx] - t_full[chg_idx][0]
                        Q_chg = abs(self.anodic_current) * t_chg / 3600 * 1000
                        Q_full = np.concatenate([Q_dis, Q_chg + Q_dis[-1]])
                        v_full_cycle = np.concatenate([v_full[dis_idx], v_full[chg_idx]])
                    else:
                        Q_full = Q_dis
                        v_full_cycle = v_full[dis_idx]

                    plot_label = "_nolegend_" if cycle in cycle_colors else label
                    ax.plot(Q_full, v_full_cycle, label=plot_label, **_cycle_plot_kwargs(cycle))
                    xlabel = "Capacity (mA·h)"
                else:
                    x_seg = t_seg;
                    xlabel = f"Time ({self.units.get('Time', 's')})"
                    plot_label = "_nolegend_" if cycle in cycle_colors else label
                    ax.plot(x_seg, v_seg, label=plot_label, **_cycle_plot_kwargs(cycle))

            # for discharge/charge/both
            for part, idxs in [('discharge', dis_idx), ('charge', chg_idx)]:
                if part == 'discharge' and seg_type not in {'discharge', 'both'}:
                    continue
                if part == 'charge' and seg_type not in {'charge', 'both'}:
                    continue
                if idxs.size == 0:
                    continue
                t_seg = t_full[idxs] - t_full[idxs][0]
                v_seg = v_full[idxs]
                # x-axis mapping
                if x_mode == 'capacity':
                    I = self.cathodic_current if part == 'discharge' else self.anodic_current
                    x_seg = abs(I) * t_seg / 3600 * 1000
                    xlabel = "Capacity (mA·h)"
                else:
                    x_seg = t_seg;
                    xlabel = f"Time ({self.units.get('Time', 's')})"
                # label
                if seg_type == 'both':
                    label = f"Cycle {cycle} {part.capitalize()}"
                else:
                    label = f"Cycle {cycle}"
                plot_label = "_nolegend_" if cycle in cycle_colors else label
                ax.plot(x_seg, v_seg, label=plot_label, **_cycle_plot_kwargs(cycle))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Potential ({self.units.get('Potential','V')})")
        title_opt = options.get('title', True)
        if title_opt:
            title, default_subtitle = _resolve_single_plot_title_subtitle(self, options)
            if title_opt in (True, "auto"):
                cycle_text = self._plot_cycles_title_text(sel_cycles)
                axis_text = "capacity axis" if x_mode == "capacity" else "time axis"
                subtitle = f"CP cycles {cycle_text}, {axis_text}"
            else:
                subtitle = default_subtitle
            title_fs = options.get("title fontsize")
            if title_fs in (None, "auto"):
                title_fs = _resolve_title_fontsize(title)
            subtitle_fs = options.get("subtitle fontsize")
            if subtitle_fs in (None, "auto"):
                subtitle_fs = _resolve_subtitle_fontsize(subtitle)
            _apply_plot_titles(fig, ax, title, subtitle, title_fs, subtitle_fs)
        if options.get('legend',False):
            legend_fs = options.get('legend fontsize') or _default_legend_fontsize()
            if gradient_enabled and str(options.get("legend mode", "auto")).strip().lower() in {"auto", "colorbar"}:
                _draw_multiplot_legend_and_colorbars(ax, color_spec, options, legend_fs)
            else:
                ax.legend(
                    fontsize=legend_fs,
                    loc=_normalize_legend_loc(options.get("legend loc", "best")),
                    bbox_to_anchor=options.get("legend bbox to anchor", None),
                )
        _apply_ecat_axis_style(ax, options)
        return ax

    @staticmethod
    def _resolve_cycles_option(cycles_opt, total_cycles):
        if cycles_opt is None:
            return list(range(1, total_cycles + 1))
        if isinstance(cycles_opt, int):
            return [cycles_opt] if 1 <= cycles_opt <= total_cycles else []
        if isinstance(cycles_opt, list):
            return [c for c in cycles_opt if isinstance(c, int) and 1 <= c <= total_cycles]
        if isinstance(cycles_opt, tuple) and len(cycles_opt) == 2:
            start, end = cycles_opt
            return [c for c in range(start, end + 1) if 1 <= c <= total_cycles]
        if isinstance(cycles_opt, tuple) and len(cycles_opt) == 3:
            start, end, step = cycles_opt
            return [c for c in range(start, end + 1, step) if 1 <= c <= total_cycles]
        raise ValueError(
            "options['cycles'] must be None, an int, a list of ints, or a tuple (start,end[,step])"
        )

    @staticmethod
    def _plot_cycles_title_text(sel_cycles):
        if not sel_cycles:
            return ""
        if len(sel_cycles) <= 5:
            return ", ".join(str(cycle) for cycle in sel_cycles)
        steps = np.diff(sel_cycles)
        if len(steps) > 0 and np.all(steps == steps[0]):
            return f"{sel_cycles[0]}-{sel_cycles[-1]} step {int(steps[0])}"
        return f"{sel_cycles[0]}-{sel_cycles[-1]}"

    def cycling_plot(
            self,
            options={},
            **mpl_kwargs
    ):
        """
        Plot cycling performance using cycle_info DataFrame.

        options may include:
          'max cycles': int     # limit number of cycles
          'percent capacity': bool
          'ma window': int      # moving average window
          'marker reference': int
          'top padding': float
          'theoretical moles': float
          'legend': bool
          'capacity mode': 'discharge', 'charge', or 'both'  # which capacity curves to show
        """
        df = self.cycle_info(options)
        maxc = options.get('max cycles')
        cycles_opt = options.get('cycles')
        if cycles_opt is not None:
            total_cycles = int(df['Cycle'].max()) if len(df) else 0
            sel_cycles = self._resolve_cycles_option(cycles_opt, total_cycles)
            df = df[df['Cycle'].isin(sel_cycles)].copy()
            if len(df):
                df["_cycle_order"] = pd.Categorical(df["Cycle"], categories=sel_cycles, ordered=True)
                df = df.sort_values("_cycle_order").drop(columns="_cycle_order")
        elif isinstance(maxc, int) and maxc > 0:
            df = df.iloc[:maxc].copy()

        cycles = df['Cycle']
        cap_col_unit = 'mA·h'

        cap_mode = options.get('capacity mode', 'both')
        if cap_mode not in ('discharge', 'charge', 'both'):
            raise ValueError("options['capacity mode'] must be 'discharge','charge', or 'both'")
        eff_mode = options.get('efficiency mode', 'both')
        if eff_mode not in ('energy', 'coulombic', 'both'):
            raise ValueError("options['capacity mode'] must be 'energy', 'coulombic', or 'both'")

        # percent capacity applies only to discharge
        if options.get('percent capacity'):
            cap_col_unit = '%'
        cap_col = f'Capacity ({cap_col_unit})'
        dis_col = f'Discharge {cap_col}'
        chg_col = f'Charge {cap_col}'
        eff_col = 'Efficiency (%)'
        EE_col = f'Energy {eff_col}'
        CE_col = f'Coulombic {eff_col}'

        # Prepare plot
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        cap_color = mpl_kwargs.get('cap_color', 'k')
        eff_color = mpl_kwargs.get('eff_color', 'tab:red')
        alpha = mpl_kwargs.get('alpha', 1)

        # Moving average
        ma = options.get('ma window')
        if ma and ma > 1:
            alpha = 0.5
            if cap_mode in ('discharge', 'both'):
                df['dis_ma'] = df[dis_col].rolling(ma).mean()
                ma_y = df['dis_ma'].dropna()
                ax1.plot(cycles[ma-1:ma-1+len(ma_y)], ma_y, '-',
                         color=cap_color, label='Discharge MA')
            if cap_mode in ('charge', 'both'):
                df['chg_ma'] = df[chg_col].rolling(ma).mean()
                ma_y = df['chg_ma'].dropna()
                ax1.plot(cycles[ma-1:ma-1+len(ma_y)], ma_y, '--',
                         color=cap_color, label='Charge MA')
            if eff_mode in ('coulombic', 'both'):
                df['CE_ma'] = df[CE_col].rolling(ma).mean()
                ma_y = df['CE_ma'].dropna()
                ax2.plot(cycles[ma-1:ma-1+len(ma_y)], ma_y, '-',
                         color=eff_color, label='CE MA')
            if eff_mode in ('energy', 'both'):
                df['EE_ma'] = df[EE_col].rolling(ma).mean()
                ma_y = df['EE_ma'].dropna()
                ax2.plot(cycles[ma-1:ma-1+len(ma_y)], ma_y, '--',
                         color=eff_color, label='EE MA')

        # Determine point size
        n_pts = len(cycles)
        ref_pts = options.get('marker reference', 10)
        default_ms = plt.rcParams.get('lines.markersize', 6) ** 2
        scale = min(1.0, ref_pts / max(n_pts, 1))
        ms = default_ms * scale ** 0.5

        # Scatter raw
        if cap_mode in ('discharge', 'both'):
            ax1.scatter(cycles, df[dis_col], color=cap_color,
                        marker='o', s=ms, linewidths=0,
                        alpha=alpha, label='Discharge')
        if cap_mode in ('charge', 'both'):
            ax1.scatter(cycles, df[chg_col], color=cap_color,
                        marker='s', s=ms, linewidths=0,
                        alpha=alpha, label='Charge')
        if eff_mode in ('coulombic', 'both'):
            ax2.scatter(cycles, df[CE_col], color=eff_color,
                        marker='x', s=ms*2,
                        alpha=alpha, label='CE')
        if eff_mode in ('energy', 'both'):
            ax2.scatter(cycles, df[EE_col], color=eff_color,
                        marker='+', s=ms*2,
                        alpha=alpha, label='EE')

        # Axes formatting
        ax1.set_xlabel('Cycle Number')

        ax1_ylabel = cap_col
        if cap_mode == 'discharge':
            ax1_ylabel = dis_col
        elif cap_mode == 'charge':
            ax1_ylabel = chg_col
        ax1.set_ylabel(ax1_ylabel, color=cap_color)
        ax1.yaxis.label.set_color(cap_color)
        ax1.tick_params(axis='y', colors=cap_color)
        ax1.spines['left'].set_color(cap_color)
        ax1.tick_params(which='minor', axis='y', colors=cap_color)

        ax2_ylabel = eff_col
        if eff_mode == 'energy':
            ax2_ylabel = EE_col
        elif eff_mode == 'coulombic':
            ax2_ylabel = CE_col
        ax2.set_ylabel(ax2_ylabel, color=eff_color)
        ax2.yaxis.label.set_color(eff_color)
        ax2.tick_params(axis='y', colors=eff_color)
        ax2.spines['right'].set_color(eff_color)
        ax2.tick_params(which='minor', axis='y', colors=eff_color)

        # Theoretical capacity line
        if 'theoretical moles' in options:
            mol = options.get('theoretical moles')
            Q_th = mol * F * 1000 / 3600
            if options.get('percent capacity') and cap_mode in ('discharge', 'both'):
                theo_val = Q_th / df[raw_dis].iloc[0] * 100
            else:
                theo_val = Q_th
            ax1.axhline(theo_val, linestyle='--', color='gray',
                        label='Theoretical')

        # Floor and pad
        if not options.get('percent capacity'):
            ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        y0, y1 = ax1.get_ylim()
        pad = (y1 - y0) * options.get('top padding', 0.1)
        ax1.set_ylim(bottom=y0, top=y1 + pad)
        y0, y1 = ax2.get_ylim()
        pad = (y1 - y0) * options.get('top padding', 0.1)
        ax2.set_ylim(bottom=y0, top=y1 + pad)

        # Legend & layout
        if options.get('legend', False):
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, **mpl_kwargs)
        _apply_ecat_axis_style(ax1, options)
        _apply_ecat_axis_style(ax2, options)
        title_opt = options.get('title', True)
        plt.tight_layout(rect=(0, 0, 1, 0.92) if title_opt else None)
        if title_opt:
            title, default_subtitle = _resolve_single_plot_title_subtitle(self, options)
            if title_opt in (True, "auto"):
                if cap_mode == "both":
                    cap_text = "capacity"
                else:
                    cap_text = f"{cap_mode} capacity"
                if eff_mode == "both":
                    eff_text = "efficiency"
                else:
                    eff_text = f"{eff_mode} efficiency"
                subtitle = f"CP cycling performance, {cap_text} and {eff_text}"
            else:
                subtitle = default_subtitle
            title_fs = options.get("title fontsize")
            if title_fs in (None, "auto"):
                title_fs = _resolve_title_fontsize(title)
            subtitle_fs = options.get("subtitle fontsize")
            if subtitle_fs in (None, "auto"):
                subtitle_fs = _resolve_subtitle_fontsize(subtitle)
            _apply_plot_titles(fig, ax1, title, subtitle, title_fs, subtitle_fs)
            if fig._suptitle is not None:
                fig._suptitle.set_y(options.get("title y", 0.98))
        return fig, (ax1, ax2)
    
    def cycling_plot_old(
        self,
        options={},
        **mpl_kwargs
    ):
        """
        Plot cycling performance using cycle_info DataFrame.
        """
        df = self.cycle_info(options)
        maxc = options.get('max cycles')
        if isinstance(maxc, int) and maxc > 0:
            df = df.iloc[:maxc].copy()

        cycles = df['Cycle']
        raw_cap_col = 'Discharge Capacity (mA·h)'
        cap_col = 'Percent Discharge Capacity (%)' if options.get('percent capacity') else raw_cap_col
        CE_col  = 'Coulombic Efficiency (%)'

        # Prepare plot
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        cap_color = mpl_kwargs.get('cap_color', 'k')
        ce_color  = mpl_kwargs.get('ce_color', 'tab:red')
        alpha = mpl_kwargs.get('alpha', 1)

        # Moving average
        ma = options.get('ma window')
        if ma and ma > 1:
            alpha = 0.5
            df['cap_ma'] = df[cap_col].rolling(ma).mean()
            df['CE_ma']  = df[CE_col].rolling(ma).mean()
            ax1.plot(cycles[ma-1:], df['cap_ma'].dropna(), '-', color=cap_color, label='Capacity Moving Avg.')
            ax2.plot(cycles[ma:], df['CE_ma'].dropna(), '-', color=ce_color, label='CE Moving Avg.')

        # Determine point size
        n_pts = len(cycles)
        ref_pts = options.get('marker reference', 10)
        default_ms = plt.rcParams.get('lines.markersize', 6) ** 2
        scale = min(1.0, ref_pts / max(n_pts, 1))
        ms = default_ms * scale**0.5

        # Scatter raw
        ax1.scatter(cycles, df[cap_col], color=cap_color, marker='o', s=ms, linewidths=0, alpha=alpha, label='Capacity')
        ax2.scatter(cycles, df[CE_col], color=ce_color, marker='o', s=ms, linewidths=0, alpha=alpha, label='CE')

        # Axes formatting
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel(cap_col, color=cap_color)
        ax1.yaxis.label.set_color(cap_color)
        ax1.tick_params(axis='y', colors=cap_color)
        ax1.spines['left'].set_color(cap_color)
        ax1.tick_params(which='minor', axis='y', colors=cap_color)

        ax2.set_ylabel(CE_col, color=ce_color)
        ax2.yaxis.label.set_color(ce_color)
        ax2.tick_params(axis='y', colors=ce_color)
        ax2.spines['right'].set_color(ce_color)
        ax2.tick_params(which='minor', axis='y', colors=ce_color)


                # Theoretical capacity line (if requested)
        if 'theoretical moles' in options:
            mol = options.get('theoretical moles')
            # compute absolute Q_th in mA·h
            Q_th = mol * F * 1000 / 3600
            if options.get('percent capacity', False):
                # baseline first-discharge for % scale
                theo_val = Q_th / df[raw_cap_col].iloc[0] * 100
            else:
                theo_val = Q_th
            ax1.axhline(theo_val, linestyle='--', color='gray', label='Theoretical Capacity')

        if not options.get('percent capacity'):
            ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)

        y0, y1 = ax1.get_ylim()
        pad = (y1 - y0) * options.get('top padding', 0.1)
        ax1.set_ylim(bottom=y0, top=y1 + pad)

        y0, y1 = ax2.get_ylim()
        pad = (y1 - y0) * options.get('top padding', 0.1)
        ax2.set_ylim(bottom=y0, top=y1 + pad)

        # Legend & layout & layout
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if options.get('legend', False):
            ax1.legend(h1+h2,l1[::-1]+l2, **mpl_kwargs)
        plt.title('Cycling Performance')
        plt.tight_layout()
        return fig, (ax1, ax2)

class ca(echem):
    """Chronoamperometry object with charge-integration helpers.
    
    Parameters
    ----------
    filepath : str or path-like
        CA text file to parse.
    options : dict or ImportOptions, optional
        Import and parser options. See ``e.describe_options("get_data")``.
    
    Examples
    --------
    >>> ca_obj = e.ca(path, {"software": "CH"})
    """
    def __init__(self, filepath, options={}):
        super().__init__(filepath, options)
        self.type = "Chronoamperometry"
        # CA-specific parameters
        self.init_E = None
        self.sample_interval = None
        self.run_time = None
        self.quiet_time = None
        self.sensitivity = None
        # parse file and load data
        self._parse_ch_ca_file(filepath)

    def _parse_ch_ca_file(self, filepath):
        """
        Parse CHI chronoamperometry (.bin, .txt) file format.
        Extract header info, then read Time/current data.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header_lines = []
        data_start = 0
        # find where data begins (line with "Time/sec")
        for idx, line in enumerate(lines):
            if re.match(r"\s*Time\s*/sec", line):
                data_start = idx
                break
            header_lines.append(line.strip())
        self._parse_ir_compensation_from_lines(header_lines)

        # extract header parameters
        for line in header_lines:
            if m := re.search(r'Init E \(V\)\s*=\s*([\d.eE+\-]+)', line):
                self.init_E = float(m.group(1))
            elif m := re.search(r'Sample Interval \(s\)\s*=\s*([\d.eE+\-]+)', line):
                self.sample_interval = float(m.group(1))
            elif m := re.search(r'Run Time \(sec\)\s*=\s*([\d.eE+\-]+)', line):
                self.run_time = float(m.group(1))
            elif m := re.search(r'Quiet Time \(sec\)\s*=\s*([\d.eE+\-]+)', line):
                self.quiet_time = float(m.group(1))
            elif m := re.search(r'Sensitivity \(A/V\)\s*=\s*([\d.eE+\-]+)', line):
                self.sensitivity = float(m.group(1))

        # read the time-current data
        df = pd.read_csv(
            filepath,
            skiprows=data_start+1,
            names=['Time', 'Current'],
            sep=',',
            comment='#'
        )
        # assign to object
        self.data = df
        # set units for plotting
        self.units['Time'] = 's'
        self.units['Current'] = 'A'

        # calculate delta_x for plotting convenience
        if len(df) >= 2:
            self.delta_x = df['Time'].iloc[1] - df['Time'].iloc[0]

    def stats(self):
        """
        Return key chronoamperometry statistics as a dict.
        """
        df = self.data
        stats = {
            'solvent': getattr(self, 'solvent', None),
            'gas': getattr(self, 'gas', None),
            'compounds': getattr(self, 'compounds', None),
            'concentrations': getattr(self, 'concentrations', None),
            'Init Potential (V)': getattr(self, 'init_E', None),
            'Sample Interval (s)': getattr(self, 'sample_interval', None),
            'Run Time (s)': getattr(self, 'run_time', None),
            'Quiet Time (s)': getattr(self, 'quiet_time', None),
            'Sensitivity (A/V)': getattr(self, 'sensitivity', None),
            'ir comp resistance': getattr(self, 'ir_comp_resistance', None),
            'ir uncomp resistance': getattr(self, 'ir_uncomp_resistance', None),
            'ir comp percent': getattr(self, 'ir_comp_percent', None),
            'Min Current (A)': df['Current'].min() if not df.empty else None,
            'Max Current (A)': df['Current'].max() if not df.empty else None,
            'Avg Current (A)': df['Current'].mean() if not df.empty else None,
        }
        return stats

    def txt_stats(self, options=None):
        if options is None:
            options = {}

        stats = {
            "exp type": _exp_type_short(getattr(self, "type", "")),
        }
        for key in [
            "Init Potential (V)",
            "Sample Interval (s)",
            "Run Time (s)",
            "Quiet Time (s)",
            "Sensitivity (A/V)",
            "Min Current (A)",
            "Max Current (A)",
            "Avg Current (A)",
        ]:
            stats.pop(key, None)

        stats["technique"] = "CA"
        stats["solvent"] = getattr(self, "solvent", None)
        stats["gas"] = getattr(self, "gas", None) or ""
        stats["compounds"] = self.combine_concs_chems(
            list(getattr(self, "concentrations", []) or []),
            list(getattr(self, "compounds", []) or []),
            options,
        )
        stats["ir comp resistance"] = getattr(self, "ir_comp_resistance", None)
        stats["ir uncomp resistance"] = getattr(self, "ir_uncomp_resistance", None)
        stats["ir comp percent"] = getattr(self, "ir_comp_percent", None)

        init_E = getattr(self, "init_E", None)
        if init_E is not None:
            stats["applied potential"] = f"{float(init_E):g} V"

        run_time = getattr(self, "run_time", None)
        if run_time is not None:
            scale, unit = scale_time_axis(np.asarray([float(run_time)]), "s", "auto")
            stats["run time"] = f"{float(run_time) * scale:g} {unit}"

        sample_interval = getattr(self, "sample_interval", None)
        if sample_interval is not None:
            stats["sample interval"] = f"{float(sample_interval):g} s"

        return stats

    def _charge_trace(self, current=None):
        i = self.data['Current'].values if current is None else np.asarray(current, dtype=float)
        t = self.data['Time'].values
        dt = np.diff(t, prepend=t[0])
        return t, np.cumsum(i * dt)

    def _resolve_baseline_correction(self, options, t, current):
        correction = options.get("baseline correction", False)
        threshold = options.get("baseline threshold", None)

        if isinstance(correction, str):
            correction = correction.strip().lower().replace("_", " ").replace("-", " ")
            if correction in {"false", "off", "none", "0"}:
                correction = False
            elif correction in {"true", "on", "1"}:
                correction = True

        if correction is True:
            mode = "threshold" if threshold is not None else "tail"
        elif correction in {"tail", "threshold"}:
            mode = correction
        elif correction:
            raise ValueError("'baseline correction' must be True, False, 'tail', or 'threshold'.")
        else:
            return None

        current = np.asarray(current, dtype=float)
        if len(current) == 0:
            return None

        if mode == "threshold":
            if threshold is None:
                raise ValueError("'baseline threshold' is required for threshold baseline correction.")
            threshold = abs(float(threshold))
            idxs = np.where(np.abs(current) <= threshold)[0]
            if len(idxs) == 0:
                raise ValueError(
                    f"No current values reach the baseline threshold ({threshold:g} A)."
                )
            idx = int(idxs[0])
            sign = np.sign(current[idx])
            if sign == 0:
                sign = np.sign(np.nanmedian(current))
            if sign == 0:
                sign = 1
            baseline_current = float(sign * threshold)
            baseline_time = float(t[idx])
            tail_fraction = None
        else:
            tail_fraction = float(options.get("baseline tail fraction", 0.05))
            n_tail = max(1, int(np.ceil(len(current) * tail_fraction)))
            baseline_current = float(np.nanmedian(current[-n_tail:]))
            baseline_time = float(t[-n_tail])
            threshold = None

        corrected_current = current - baseline_current
        _, corrected_charge = self._charge_trace(corrected_current)
        _, removed_charge = self._charge_trace(np.full_like(current, baseline_current, dtype=float))
        return {
            "mode": mode,
            "baseline current": baseline_current,
            "baseline time": baseline_time,
            "baseline threshold": threshold,
            "baseline tail fraction": tail_fraction,
            "corrected current": corrected_current,
            "corrected charge": corrected_charge,
            "removed charge": removed_charge,
        }

    def _resolve_charge_target(self, options, charge=None):
        target_charge = options.get("target charge", None)
        if charge is not None:
            target_charge = charge

        target_moles = options.get("target moles", None)
        target_electrons = options.get("target electrons", None)

        if target_charge is not None and target_moles is not None:
            raise ValueError("Use either 'target charge' or 'target moles', not both.")

        if target_charge is None and target_moles is not None:
            if target_electrons is None:
                raise ValueError("'target moles' requires 'target electrons'.")
            target_charge = float(target_moles) * float(target_electrons) * F

        return None if target_charge is None else float(target_charge)

    def _time_at_charge_value(self, target_charge, t, Q):
        q_lo, q_hi = np.nanmin(Q), np.nanmax(Q)
        if not (q_lo <= target_charge <= q_hi):
            raise ValueError(
                f"Requested charge ({target_charge:g} C) is outside the data range "
                f"({q_lo:g} – {q_hi:g} C)."
            )
        return float(np.interp(target_charge, Q, t))

    def _plot_charge_trace(self, t, Q, options, **mpl_kwargs):
        x_unit = self.units.get('Time', 's')
        selected = options.get('x unit', 'auto')
        scale, x_unit_lbl = scale_time_axis(t, x_unit, selected)
        t_plot = t * scale
        ax = plt.gca()
        if "color" not in mpl_kwargs:
            mpl_kwargs["color"] = options.get("color", "black")
        ax.plot(t_plot, Q, **mpl_kwargs)
        ax.set_xlabel(f'Time {x_unit_lbl}')
        ax.set_ylabel('Charge (C)')
        _apply_ecat_axis_style(ax, options)
        _add_scale_bar(ax, options, unit="C")
        return ax, t_plot, x_unit_lbl

    def _plot_charge_target(self, ax, target_charge, t_at_plot, options, *, charge_axis=True):
        color = options.get("charge color", "tab:red")
        label = options.get("target label")
        if label is None:
            label = f"{target_charge:g} C"
        if charge_axis:
            ax.axhline(target_charge, ls="--", color=color, label=label)
        ax.axvline(t_at_plot, ls="--", color=color)
        if options.get("legend", False) is True:
            ax.legend()

    def charge(self, options={}, **mpl_kwargs):
        """Integrate CA current to cumulative charge and optionally plot it.
        
        Parameters
        ----------
        options : dict or PlotOptions, optional
            Charge display and target options. See ``e.describe_options("ca.charge")``.
        **mpl_kwargs
            Additional keyword arguments passed to Matplotlib.
        
        Returns
        -------
        ChronoAnalysisResult
            Dict-like result with cumulative charge, final charge, optional target
            charge, target time, and plotted axes.
        
        Examples
        --------
        >>> result = ca_obj.charge({"plot": False})
        >>> result["final charge"]
        """
        options = PlotOptions.from_options(options).to_legacy_dict()
        options.update(mpl_kwargs)

        t, Q = self._charge_trace()
        current = self.data['Current'].values
        baseline = self._resolve_baseline_correction(options, t, current)
        analysis_Q = baseline["corrected charge"] if baseline is not None else Q
        target_charge = self._resolve_charge_target(options)
        t_at = None
        ax = None

        if target_charge is not None:
            t_at = self._time_at_charge_value(target_charge, t, analysis_Q)

        if options.get("plot", True):
            ax, t_plot, x_unit_lbl = self._plot_charge_trace(t, analysis_Q, options)
            if target_charge is not None and options.get("plot target", True):
                t_at_plot = float(np.interp(target_charge, analysis_Q, t_plot))
                self._plot_charge_target(ax, target_charge, t_at_plot, options, charge_axis=True)
            title_opt = options.get('title', True)
            title, subtitle = _resolve_single_plot_title_subtitle(self, options)
            title_fs = options.get("title fontsize")
            if title_fs in (None, "auto"):
                title_fs = _resolve_title_fontsize(title)
            subtitle_fs = options.get("subtitle fontsize")
            if subtitle_fs in (None, "auto"):
                subtitle_fs = _resolve_subtitle_fontsize(subtitle)
            if title_opt:
                _apply_plot_titles(ax.figure, ax, title, subtitle, title_fs, subtitle_fs)
        else:
            x_unit_lbl = self.units.get("Time", "s")

        if options.get("print", True):
            print(f"Final charge: {round_sigfigs(float(Q[-1]), options.get('sig figs', 4))} C")
            if target_charge is not None and t_at is not None:
                pretty_t = round_sigfigs(t_at, options.get("sig figs", 4))
                print(f"t({target_charge:g} C) = {pretty_t} {self.units.get('Time', 's')}")

        values = {
            "time": pd.Series(t, name="Time"),
            "charge": pd.Series(Q, name="Charge"),
            "final charge": float(Q[-1]) if len(Q) else np.nan,
            "target charge": target_charge,
            "time at target charge": t_at,
            "time unit": self.units.get("Time", "s"),
            "display time unit": x_unit_lbl,
        }
        if baseline is not None:
            values.update({
                "baseline correction": baseline["mode"],
                "baseline current": baseline["baseline current"],
                "baseline time": baseline["baseline time"],
                "baseline threshold": baseline["baseline threshold"],
                "baseline tail fraction": baseline["baseline tail fraction"],
                "corrected current": pd.Series(baseline["corrected current"], name="Corrected Current"),
                "corrected charge": pd.Series(baseline["corrected charge"], name="Corrected Charge"),
                "final corrected charge": float(baseline["corrected charge"][-1]) if len(Q) else np.nan,
                "removed charge": pd.Series(baseline["removed charge"], name="Removed Charge"),
            })
        return ChronoAnalysisResult(values, axes=ax)

    def plot(self, options={}, **mpl_kwargs):
        """Plot chronoamperometry current versus time.
        
        Parameters
        ----------
        options : dict or PlotOptions, optional
            Axis, scaling, title, and legend options. See ``e.describe_options("plot")``.
        **mpl_kwargs
            Additional keyword arguments passed to Matplotlib.
        
        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the CA trace.
        
        Examples
        --------
        >>> ca_obj.plot({"y axis": "current"})
        """
        options = {} if options is None else dict(options)
        if (
            options.get("plot charge", False)
            and "color" not in options
            and "color" not in mpl_kwargs
        ):
            options["color"] = "black"

        # extract data
        t = self.data['Time'].values
        i = self.data['Current'].values

        # plot current vs. time
        ax1 = super().plot(options, **mpl_kwargs)
        y0, y1 = ax1.get_ylim()
        if y0 > 0:
            ax1.set_ylim(bottom=0)
        elif y1 < 0:
            ax1.set_ylim(top=0)
        #ax1.set_xlabel(f"Time ({self.units.get('Time','s')})")
        #ax1.set_ylabel(f"Current ({self.units.get('Current','A')})")

        # optional cumulative charge curve
        if options.get('plot charge', False):
            x_unit = self.units.get('Time', 's')
            selected = options.get('x unit', 'auto')
            scale, _ = scale_time_axis(t, x_unit, selected)
            t_plot = t * scale
            _, Q = self._charge_trace()
            baseline = self._resolve_baseline_correction(options, t, i)
            if baseline is not None:
                Q = baseline["corrected charge"]
            ax2 = ax1.twinx()
            charge_color = options.get('charge color', 'tab:red')
            ax2.plot(t_plot, Q, color=charge_color, label='Charge')
            ax2.set_ylabel('Charge (C)', color=charge_color)
            ax2.spines['right'].set_color(charge_color)
            ax2.tick_params(axis='y', colors=charge_color)
            ax2.tick_params(which='minor', axis='y', colors=charge_color)

            # combined legend
            if options.get('legend', False):
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2)

            return ax1
        else:
            if options.get('legend', False):
                ax1.legend()

        return ax1

    def time_at_charge(self, charge=None, options={}):
        """Find the time at which cumulative CA charge reaches a target.
        
        Parameters
        ----------
        charge : float or dict, optional
            Target charge in Coulombs, or an options dictionary with ``target charge``
            or ``target moles`` / ``target electrons``.
        options : dict, optional
            Display and target options. See ``e.describe_options("ca.time_at_charge")``.
        
        Returns
        -------
        ChronoAnalysisResult
            Dict-like result with target charge, time, and optional plotted axes.
        
        Examples
        --------
        >>> result = ca_obj.time_at_charge({"target charge": 0.001})
        >>> result["time"]
        """
        if isinstance(charge, dict) and options == {}:
            options = charge
            charge = None
        options = PlotOptions.from_options(options).to_legacy_dict()

        target_charge = self._resolve_charge_target(options, charge=charge)
        if target_charge is None:
            raise ValueError(
                "Please provide 'target charge' or 'target moles' with 'target electrons'."
            )

        t, Q = self._charge_trace()
        t_at = self._time_at_charge_value(target_charge, t, Q)
        x_unit_raw = self.units.get('Time', 's')
        selected = options.get('x unit', 'auto')
        scale, x_unit_lbl = scale_time_axis(t, x_unit_raw, selected)
        t_scaled = t * scale

        t_at_plot = float(np.interp(target_charge, Q, t_scaled))

        sig_figs = options.get('sig figs', 4)
        if options.get("print", True):
            pretty_t = round_sigfigs(t_at_plot, sig_figs)
            unit_lbl = x_unit_lbl or x_unit_raw
            print(f"t({target_charge:g} C) = {pretty_t} {unit_lbl}")

        ax = None
        if options.get("plot", False):
            if options.get("plot ca", True):
                plot_options = dict(options)
                plot_options["plot charge"] = True
                if plot_options.get("legend") is not True:
                    plot_options["legend"] = False
                ax = self.plot(plot_options)
                self._plot_charge_target(
                    ax,
                    target_charge,
                    t_at_plot,
                    options,
                    charge_axis=False,
                )
            else:
                ax, _t_plot, _x_unit_lbl = self._plot_charge_trace(t, Q, options)
                self._plot_charge_target(
                    ax,
                    target_charge,
                    t_at_plot,
                    options,
                    charge_axis=True,
                )

        return ChronoAnalysisResult(
            {
                "time": t_at,
                "display time": t_at_plot,
                "target charge": target_charge,
                "time unit": x_unit_raw,
                "display time unit": x_unit_lbl,
            },
            axes=ax,
        )

def _apply_current_scale(*args, **kwargs):
    from .analysis_cv import _apply_current_scale as impl
    return impl(*args, **kwargs)


def _apply_normalized_current_axis(*args, **kwargs):
    from .analysis_cv import _apply_normalized_current_axis as impl
    return impl(*args, **kwargs)


def _default_normalized_axis(*args, **kwargs):
    from .analysis_cv import _default_normalized_axis as impl
    return impl(*args, **kwargs)


def _find_column_by_text(*args, **kwargs):
    from .analysis_cv import _find_column_by_text as impl
    return impl(*args, **kwargs)


def _is_ip0_y_axis(*args, **kwargs):
    from .analysis_cv import _is_ip0_y_axis as impl
    return impl(*args, **kwargs)


def _normalize_single_cv(*args, **kwargs):
    from .analysis_cv import _normalize_single_cv as impl
    return impl(*args, **kwargs)


def _resolve_ip0_values(*args, **kwargs):
    from .analysis_cv import _resolve_ip0_values as impl
    return impl(*args, **kwargs)


def plateau_current(*args, **kwargs):
    from .analysis_batch import plateau_current as impl
    return impl(*args, **kwargs)


from .plotting import (
    _add_scale_bar,
    _apply_ecat_axis_style,
    _apply_plot_titles,
    _default_legend_fontsize,
    _draw_multiplot_legend_and_colorbars,
    _format_reference_label_mathtext,
    _get_group_cmap,
    _normalize_legend_loc,
    _plot_options_from_mapping,
    _resolve_adaptive_legend_layout,
    _resolve_single_plot_title_subtitle,
    _resolve_subtitle_fontsize,
    _resolve_title_fontsize,
)
from .reference import midpoint_potential

__all__ = ["echem", "cv", "ca", "cp", "dpv"]
