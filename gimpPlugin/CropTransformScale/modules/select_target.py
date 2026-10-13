#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi

gi.require_version("GimpUi", "3.0")
from gi.repository import GimpUi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

GimpUi.init("CropTransformScale")


def select_target():
    formats = [
        ("ID_CARD", "ID Card"),
        ("LONG_ID_CARD", "Long ID Card"),
        ("A5", "A5"),
        ("A4", "A4"),
        ("LEGAL", "Legal"),
        ("A3", "A3"),
        ("PASSPORT_PAGE", "Single page of a Passport"),
        ("BB_FULL", "BB Full"),
        ("BB_HALF", "BB Half"),
    ]

    pairing_formats = {"ID_CARD", "LONG_ID_CARD", "PASSPORT_PAGE"}

    dialog = GimpUi.Dialog(title="Select Image and Naming Format")
    combo = Gtk.ComboBoxText()
    for format_id, label in formats:
        combo.append(format_id, label)
    combo.set_active(0)
    content = dialog.get_content_area()
    content.set_spacing(8)
    content.add(combo)

    # Pair / Single selection
    type_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
    pair_radio = Gtk.RadioButton.new_with_label_from_widget(None, "Pair")
    single_radio = Gtk.RadioButton.new_with_label_from_widget(pair_radio, "Single")
    pair_radio.set_active(True)
    type_box.pack_start(pair_radio, False, False, 0)
    type_box.pack_start(single_radio, False, False, 0)
    content.add(type_box)

    # Group Number
    group_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    group_label = Gtk.Label(label="Group Number")
    group_entry = Gtk.Entry()
    group_entry.set_input_purpose(Gtk.InputPurpose.DIGITS)
    group_entry.set_placeholder_text("e.g.: 1")
    group_box.pack_start(group_label, False, False, 0)
    group_box.pack_start(group_entry, True, True, 0)
    content.add(group_box)

    # Position within the group
    position_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    position_label = Gtk.Label(label="Position in Group")
    position_entry = Gtk.Entry()
    position_entry.set_input_purpose(Gtk.InputPurpose.DIGITS)
    position_entry.set_placeholder_text("e.g: 1 or 2")
    position_box.pack_start(position_label, False, False, 0)
    position_box.pack_start(position_entry, True, True, 0)
    content.add(position_box)

    def update_controls(*args):
        pairing_enabled = combo.get_active_id() in pairing_formats

        type_box.set_visible(pairing_enabled)
        group_box.set_visible(pairing_enabled)
        position_box.set_visible(pairing_enabled)

        is_pair = pair_radio.get_active()
        group_box.set_sensitive(pairing_enabled and is_pair)
        position_box.set_sensitive(pairing_enabled and is_pair)

        if not pairing_enabled or not is_pair:
            group_entry.set_text("")
            position_entry.set_text("")

    combo.connect("changed", update_controls)
    pair_radio.connect("toggled", update_controls)
    single_radio.connect("toggled", update_controls)

    dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
    dialog.add_button("OK", Gtk.ResponseType.OK)

    dialog.show_all()
    update_controls()

    # Validate the numeric fields when 'Pair' is selected
    while True:
        response = dialog.run()
        if response != Gtk.ResponseType.OK:
            dialog.destroy()
            return None

        selected_format = combo.get_active_id()
        # if selected_format is None:
        #     continue
        image_type = None
        group_number = None
        position = None

        if selected_format in pairing_formats:
            if pair_radio.get_active():
                group_text = group_entry.get_text().strip()
                position_text = position_entry.get_text().strip()

                if (
                    not group_text.isdigit()
                    or int(group_text) < 1
                    or not position_text.isdigit()
                    or int(position_text) < 1
                ):
                    message = Gtk.MessageDialog(
                        transient_for=dialog,
                        flags=Gtk.DialogFlags.MODAL,
                        message_type=Gtk.MessageType.ERROR,
                        buttons=Gtk.ButtonsType.OK,
                        text="Enter a positive Group Number and Position!",
                    )
                    message.run()
                    message.destroy()
                    continue

                image_type = "P"
                group_number = int(group_text)
                position = int(position_text)

            else:
                image_type = "S"

        dialog.destroy()
        return (
            selected_format,
            image_type,
            group_number,
            position,
        )


def build_export_name(selected_format, image_type, group_number, position):
    if image_type == "P":
        return f"G{group_number}P{position}_{selected_format}"
    return selected_format


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
