from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = (
    PROJECT_ROOT
    / "data"
    / "logs"
)


def safe_text(element, default=""):
    if element is None:
        return default

    return element.text or default


def load_xml_log_documents(max_files=None):

    xml_files = sorted(
        LOG_DIR.glob("*.xml")
    )

    if max_files is not None:
        xml_files = xml_files[:max_files]

    documents = []

    print("Reading XML logs from:")
    print(LOG_DIR)

    for file_path in xml_files:

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

        except Exception as exc:

            print(
                f"WARNING: Could not parse "
                f"{file_path.name}: {exc}"
            )

            continue

        # ------------------------------------------
        # Main execution attributes
        # ------------------------------------------

        execution_id = root.attrib.get(
            "executionId",
            file_path.stem
        )

        job_name = root.attrib.get(
            "jobName",
            ""
        )

        site = root.attrib.get(
            "site",
            ""
        )

        country = root.attrib.get(
            "country",
            ""
        )

        application = root.attrib.get(
            "application",
            ""
        )

        status = root.attrib.get(
            "status",
            ""
        )

        start_time = root.attrib.get(
            "startTime",
            ""
        )

        end_time = root.attrib.get(
            "endTime",
            ""
        )

        incident_ref = root.attrib.get(
            "incidentRef",
            ""
        )

        # ------------------------------------------
        # Metadata
        # ------------------------------------------

        metadata_element = root.find(
            "metadata"
        )

        module = ""
        host = ""
        environment = ""
        version = ""

        if metadata_element is not None:

            module = safe_text(
                metadata_element.find("module")
            )

            host = safe_text(
                metadata_element.find("host")
            )

            environment = safe_text(
                metadata_element.find(
                    "environment"
                )
            )

            version = safe_text(
                metadata_element.find("version")
            )

        # ------------------------------------------
        # Failure information
        # ------------------------------------------

        failure = root.find("failure")

        error_code = ""
        error_category = ""
        symptom = ""
        probable_root_cause = ""

        if failure is not None:

            error_code = safe_text(
                failure.find("errorCode")
            )

            error_category = safe_text(
                failure.find(
                    "errorCategory"
                )
            )

            symptom = safe_text(
                failure.find("symptom")
            )

            probable_root_cause = safe_text(
                failure.find(
                    "probableRootCause"
                )
            )

        # ------------------------------------------
        # Extract log events
        # ------------------------------------------

        all_events = []
        important_events = []

        events_element = root.find(
            "events"
        )

        if events_element is not None:

            for event in events_element.findall(
                    "event"
            ):

                timestamp = event.attrib.get(
                    "timestamp",
                    ""
                )

                level = event.attrib.get(
                    "level",
                    ""
                )

                code = event.attrib.get(
                    "code",
                    ""
                )

                message = (
                    event.text or ""
                ).strip()

                event_string = (
                    f"{timestamp} | "
                    f"{level} | "
                    f"{code} | "
                    f"{message}"
                )

                all_events.append(
                    event_string
                )

                if level.upper() in [
                    "ERROR",
                    "WARN"
                ]:

                    important_events.append(
                        event_string
                    )

        # ------------------------------------------
        # Keep useful surrounding events too
        # ------------------------------------------

        recent_events = all_events[-10:]

        event_section = "\n".join(
            important_events
            + recent_events
        )

        # Remove duplicate lines
        event_lines = []

        seen = set()

        for line in event_section.splitlines():

            if line not in seen:

                seen.add(line)

                event_lines.append(
                    line
                )

        event_section = "\n".join(
            event_lines
        )

        # ------------------------------------------
        # Searchable document
        # ------------------------------------------

        content = f"""
Control-M / Application Execution Log

Execution ID:
{execution_id}

Job Name:
{job_name}

Status:
{status}

Application:
{application}

Module:
{module}

Site:
{site}

Country:
{country}

Environment:
{environment}

Host:
{host}

Version:
{version}

Start Time:
{start_time}

End Time:
{end_time}

Related Incident:
{incident_ref}

Error Code:
{error_code}

Error Category:
{error_category}

Observed Symptom:
{symptom}

Probable Root Cause:
{probable_root_cause}

Important Log Events:
{event_section}
""".strip()

        metadata = {

            "job_name": job_name,

            "status": status,

            "country": country,

            "module": module,

            "host": host,

            "environment": environment,

            "version": version,

            "start_time": start_time,

            "end_time": end_time,

            "incident_ref": incident_ref,

            "error_category":
                error_category,

            "file_name":
                file_path.name
        }

        document = {

            "source_type":
                "LOG",

            "source_id":
                execution_id,

            "chunk_index":
                0,

            "title":
                (
                    f"{job_name} - "
                    f"{status} - "
                    f"{site}"
                ),

            "content":
                content,

            "site":
                site,

            "application":
                application,

            "error_code":
                error_code,

            "metadata":
                metadata
        }

        documents.append(
            document
        )

    print(
        f"\nProcessed "
        f"{len(documents):,} "
        f"XML log files."
    )

    return documents


if __name__ == "__main__":

    docs = load_xml_log_documents(
        max_files=5
    )

    print(
        "\nSample XML log document:\n"
    )

    print(
        docs[0]
    )