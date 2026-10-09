#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi

gi.require_version("GimpUi", "3.0")
from gi.repository import GimpUi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk


def select_target():
    formats = [
        ("ID_CARD", "ID Card"),
        ("LONG_ID_CARD", "Long ID Card"),
        ("A5", "A5"),
        ("A4", "A4"),
        ("LEGAL", "Legal"),
        ("A3", "A3"),
        ("PASSPORT_PAGE", "Single page of a Passport"),
    ]
    dialog = GimpUi.Dialog(title="Select Format")
    combo = Gtk.ComboBoxText()
    for format_id, label in formats:
        combo.append(format_id, label)
    combo.set_active(0)
    content = dialog.get_content_area()
    content.add(combo)
    dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
    dialog.add_button("OK", Gtk.ResponseType.OK)
    dialog.show_all()
    response = dialog.run()
    selected = None
    if response == Gtk.ResponseType.OK:
        selected = combo.get_active_id()
    dialog.destroy()
    return selected


def confirm_export():
    dialog = GimpUi.Dialog(title="Confirm Export")
    content = dialog.get_content_area()
    label = Gtk.Label(label="Do you want to export the processed image?")
    content.add(label)
    dialog.add_button("No", Gtk.ResponseType.NO)
    dialog.add_button("Yes", Gtk.ResponseType.YES)
    dialog.show_all()
    response = dialog.run()
    dialog.destroy()
    return response == Gtk.ResponseType.YES
