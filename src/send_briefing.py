#!/usr/bin/env python3
"""Send a finished briefing via Gmail SMTP.

This is the delivery step for the Claude Routine architecture (see ROUTINE.md).
Each scheduled run, Claude researches and writes the briefing as markdown, then
calls this script to convert it to HTML and email it. The script is intentionally
small and dependency-light (stdlib + `markdown`) so it runs anywhere the routine
environment is set up.

Usage:
    python src/send_briefing.py --input briefing.md
    python src/send_briefing.py --input briefing.md --subject "Public Sector Daily — 2026-06-15"
    cat briefing.md | python src/send_briefing.py

Required environment variables (set them on the routine's cloud environment):
    SMTP_USER       Gmail sender address
    SMTP_PASSWORD   Gmail App Password (requires 2FA; not the account password)
    RECIPIENT_EMAIL Recipient inbox(es), comma-separated for multiple

Optional:
    SUBJECT_PREFIX  Defaults to "Public Sector Daily"
"""
from __future__ import annotations

import argparse
import datetime as dt
import html as html_lib
import os
import smtplib
import ssl
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

import markdown as md


def require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise SystemExit(
            f"ERROR: required environment variable {name} is not set. "
            "Configure it on the routine's cloud environment before sending."
        )
    return value


def parse_recipients(raw: str) -> List[str]:
    recipients = [part.strip() for part in raw.replace(";", ",").split(",")]
    recipients = [r for r in recipients if r]
    if not recipients:
        raise SystemExit("ERROR: RECIPIENT_EMAIL did not contain any usable addresses.")
    return recipients


def read_markdown(input_path: Optional[str]) -> str:
    if input_path:
        with open(input_path, "r", encoding="utf-8") as handle:
            text = handle.read()
    else:
        text = sys.stdin.read()
    text = (text or "").strip()
    if not text:
        raise SystemExit("ERROR: no briefing content provided (empty input).")
    return text


def markdown_to_html(markdown_text: str, footer_note: Optional[str] = None) -> str:
    body = md.markdown(
        markdown_text,
        extensions=["extra", "sane_lists", "smarty"],
        output_format="html5",
    )
    footer_html = ""
    if footer_note:
        footer_html = (
            '\n    <hr style="border:none;border-top:1px solid #ddd;margin:24px 0;"/>'
            f'\n    <p style="font-size:12px;color:#888;line-height:1.5;">{html_lib.escape(footer_note)}</p>'
        )
    return f"""\
<!DOCTYPE html>
<html lang="de">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
    <title>Public Sector Daily</title>
  </head>
  <body style="font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.55;color:#1a1a1a;max-width:800px;margin:0 auto;padding:16px 20px;">
    {body}
    {footer_html}
  </body>
</html>
"""


def send_email(subject: str, html_body: str, sender: str, recipients: List[str], smtp_password: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30, context=context) as server:
            server.login(sender, smtp_password)
            server.sendmail(sender, recipients, msg.as_string())
    except smtplib.SMTPAuthenticationError as exc:
        raise SystemExit(
            "ERROR: Gmail SMTP authentication failed. Verify SMTP_USER and SMTP_PASSWORD "
            "(must be a Gmail App Password with 2FA enabled, not your account password). "
            f"Detail: {exc}"
        ) from exc
    except smtplib.SMTPRecipientsRefused as exc:
        raise SystemExit(f"ERROR: SMTP rejected all recipients {recipients}: {exc}") from exc
    except smtplib.SMTPException as exc:
        raise SystemExit(f"ERROR: SMTP error sending to {recipients}: {exc}") from exc


def default_subject(prefix: str) -> str:
    today = dt.date.today().isoformat()
    return f"{prefix} — {today}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a markdown briefing via Gmail SMTP.")
    parser.add_argument("--input", "-i", default=None, help="Path to the briefing markdown file. Reads stdin if omitted.")
    parser.add_argument("--subject", "-s", default=None, help="Full email subject. Overrides --subject-prefix.")
    parser.add_argument(
        "--subject-prefix",
        default=os.environ.get("SUBJECT_PREFIX", "Public Sector Daily"),
        help="Subject prefix used to build a dated subject when --subject is not given.",
    )
    parser.add_argument("--footer", default=None, help="Optional small footer note appended below the briefing.")
    parser.add_argument("--dry-run", action="store_true", help="Render and print the HTML without sending.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    markdown_text = read_markdown(args.input)
    subject = args.subject or default_subject(args.subject_prefix)
    html_body = markdown_to_html(markdown_text, footer_note=args.footer)

    if args.dry_run:
        print(f"INFO: dry run — subject: {subject}")
        print(html_body)
        return 0

    sender = require_env("SMTP_USER")
    smtp_password = require_env("SMTP_PASSWORD")
    recipients = parse_recipients(require_env("RECIPIENT_EMAIL"))

    send_email(subject, html_body, sender, recipients, smtp_password)
    print(f"INFO: briefing sent to {len(recipients)} recipient(s): {', '.join(recipients)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
