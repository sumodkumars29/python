import bpy

# ==========================================================
# Configuration
# ==========================================================

TARGET_NAME = "Store Named Attribute.009"

GENERIC_SOCKET_NAMES = {
    "",
    "Geometry",
    None,
}

INSTANCE_CREATORS = {
    "GeometryNodeInstanceOnPoints",
    "GeometryNodeInstanceOnElements",
}

REALIZE_INSTANCES = "GeometryNodeRealizeInstances"


# ==========================================================
# Basic helpers
# ==========================================================


def is_sna(node):
    return node.bl_idname == "GeometryNodeStoreNamedAttribute"


def get_geometry_input(node):
    for socket in node.inputs:
        if socket.bl_idname == "NodeSocketGeometry":
            return socket

    return None


def get_geometry_output(node):
    for socket in node.outputs:
        if socket.bl_idname == "NodeSocketGeometry":
            return socket

    return None


def useful_socket_name(socket):
    if not socket:
        return None

    if socket.name in GENERIC_SOCKET_NAMES:
        return None

    return socket.name


def find_geometry_input(node, name):
    for socket in node.inputs:
        if socket.name == name:
            return socket

    return None


# ==========================================================
# Inspect the boundary between two nodes
# ==========================================================


def inspect_connection(output_socket, input_socket):
    """
    Inspect both sides of a geometry connection.

    Returns a useful socket name if one exists.
    """

    print("    Output socket :", output_socket.name)

    clue = useful_socket_name(output_socket)

    if clue:
        print("    FOUND CLUE    :", clue)
        return clue

    print("    Input socket  :", input_socket.name)

    clue = useful_socket_name(input_socket)

    if clue:
        print("    FOUND CLUE    :", clue)
        return clue

    return None


# ==========================================================
# Find the node upstream of a geometry input
# ==========================================================


def get_upstream_geometry_node(node):

    geometry_input = get_geometry_input(node)

    if not geometry_input:
        return None, None

    if not geometry_input.is_linked:
        return None, geometry_input

    link = geometry_input.links[0]

    return link.from_node, geometry_input


# ==========================================================
# Instance handling
# ==========================================================


def is_realize_instances(node):
    return node.bl_idname == REALIZE_INSTANCES


def is_instance_creator(node):
    return node.bl_idname in INSTANCE_CREATORS


def get_instance_input(node):

    for socket in node.inputs:
        if socket.name == "Instance":
            return socket

    return None


def get_geometry_input_socket(node):

    for socket in node.inputs:
        if socket.bl_idname == "NodeSocketGeometry":
            return socket

    return None


def trace_instance_source(instance_creator):
    """
    Once an Instance on Points / Instance on Elements node
    is found, switch from its output path to its Instance input.

    The geometry connected to that input is the geometry that
    becomes the instance.
    """

    instance_input = get_instance_input(instance_creator)

    if not instance_input:
        print("    No Instance input found.")
        return None

    print()
    print("    INSTANCE INPUT")
    print("    -------------")
    print("    Socket:", instance_input.name)

    if not instance_input.is_linked:
        print("    Instance input is not linked.")
        return None

    link = instance_input.links[0]

    source_node = link.from_node
    source_socket = link.from_socket

    print("    Source node  :", source_node.name)
    print("    Source output:", source_socket.name)

    clue = useful_socket_name(source_socket)

    if clue:
        print("    FOUND CLUE:", clue)
        return clue

    return trace_geometry_backward_from(source_node)


# ==========================================================
# Continue tracing backwards from a geometry node
# ==========================================================


def trace_geometry_backward_from(start_node):

    current = start_node

    while True:

        print()
        print("    Tracing:", current.name)

        # --------------------------------------------------
        # If we encounter another instance creator,
        # switch to its Instance input.
        # --------------------------------------------------

        if is_instance_creator(current):

            print("    INSTANCE CREATOR:", current.name)

            clue = trace_instance_source(current)

            if clue:
                return clue

            return None

        # --------------------------------------------------
        # Normal geometry traversal.
        # --------------------------------------------------

        geometry_input = get_geometry_input_socket(current)

        if not geometry_input:
            print("    No Geometry input.")
            return None

        if not geometry_input.is_linked:
            print("    Geometry input is not linked.")
            return None

        link = geometry_input.links[0]

        previous_node = link.from_node
        output_socket = link.from_socket

        print("    Previous node :", previous_node.name)
        print("    Output socket :", output_socket.name)

        clue = useful_socket_name(output_socket)

        if clue:
            print("    FOUND CLUE:", clue)
            return clue

        current = previous_node


# ==========================================================
# FORWARD SEARCH
#
# Maximum TWO non-SNA nodes.
# SNA nodes are transparent and do not count.
# ==========================================================


def forward_search(target):

    print()
    print("FORWARD SEARCH")
    print("-" * 60)

    current = target
    non_sna_count = 0

    while non_sna_count < 2:

        geometry_output = get_geometry_output(current)

        if not geometry_output:
            print("No Geometry output.")
            return None

        if not geometry_output.is_linked:
            print("Geometry output is not linked.")
            return None

        link = geometry_output.links[0]

        next_node = link.to_node
        receiving_input = link.to_socket

        print()
        print("Node:", next_node.name)
        print("Is SNA:", is_sna(next_node))

        # --------------------------------------------------
        # SNA is transparent.
        # It does NOT count toward the two-node limit.
        # --------------------------------------------------

        if is_sna(next_node):

            current = next_node
            continue

        # --------------------------------------------------
        # We have found the next NON-SNA node.
        # This counts.
        # --------------------------------------------------

        non_sna_count += 1

        print("Non-SNA number:", non_sna_count)

        clue = inspect_connection(
            get_geometry_output(current),
            receiving_input,
        )

        if clue:
            return clue

        # --------------------------------------------------
        # No clue from this non-SNA.
        # Try its own geometry output as well.
        # --------------------------------------------------

        node_output = get_geometry_output(next_node)

        if node_output:

            print("    Node output:", node_output.name)

            clue = useful_socket_name(node_output)

            if clue:
                print("    FOUND CLUE:", clue)
                return clue

        current = next_node

    print()
    print("Two non-SNA nodes checked.")
    print("No useful forward clue found.")

    return None


# ==========================================================
# BACKWARD SEARCH
#
# No artificial loop limit.
#
# SNA nodes are transparent.
#
# Realize Instances is treated specially:
# continue backward until the node which creates the
# instances is found.
# ==========================================================


def backward_search(target):

    print()
    print("BACKWARD SEARCH")
    print("-" * 60)

    current = target

    while True:

        previous_node, geometry_input = get_upstream_geometry_node(current)

        if not previous_node:
            print("No upstream Geometry node.")
            return None

        print()
        print("Node:", previous_node.name)
        print("Is SNA:", is_sna(previous_node))

        # --------------------------------------------------
        # SNA is transparent.
        # --------------------------------------------------

        if is_sna(previous_node):

            current = previous_node
            continue

        # --------------------------------------------------
        # REALIZE INSTANCES
        #
        # We must keep travelling backward until we find
        # the node that actually created the instances.
        # --------------------------------------------------

        if is_realize_instances(previous_node):

            print("REALIZE INSTANCES encountered.")

            current = previous_node

            while True:

                upstream_node, upstream_input = get_upstream_geometry_node(current)

                if not upstream_node:
                    print("Could not trace beyond Realize Instances.")
                    return None

                print()
                print("Through Realize Instances ->", upstream_node.name)
                print("Is SNA:", is_sna(upstream_node))

                if is_sna(upstream_node):

                    current = upstream_node
                    continue

                # ------------------------------------------
                # Instance creator found.
                # ------------------------------------------

                if is_instance_creator(upstream_node):

                    print("INSTANCE CREATOR FOUND:", upstream_node.name)

                    return trace_instance_source(upstream_node)

                # ------------------------------------------
                # Not yet at an instance creator.
                # Continue backwards through Geometry.
                # ------------------------------------------

                output_socket = get_geometry_output(upstream_node)

                if output_socket:

                    clue = useful_socket_name(output_socket)

                    if clue:
                        print("FOUND CLUE:", clue)
                        return clue

                current = upstream_node

        # --------------------------------------------------
        # Normal NON-SNA node.
        #
        # Inspect the connection on both sides.
        # --------------------------------------------------

        print("Output socket:", geometry_input.links[0].from_socket.name)

        output_socket = geometry_input.links[0].from_socket

        clue = useful_socket_name(output_socket)

        if clue:
            print("FOUND CLUE:", clue)
            return clue

        print("Receiving input:", geometry_input.name)

        clue = useful_socket_name(geometry_input)

        if clue:
            print("FOUND CLUE:", clue)
            return clue

        # --------------------------------------------------
        # No useful clue.
        # Continue backward.
        # --------------------------------------------------

        print("No useful clue. Continuing backward.")

        current = previous_node


# ==========================================================
# Main test
# ==========================================================


def run():

    node_group = bpy.context.object.modifiers.active.node_group

    target = node_group.nodes.get(TARGET_NAME)

    if not target:
        print()
        print("TARGET NOT FOUND:", TARGET_NAME)
        return

    print()
    print("TARGET:", target.name)
    print("SNA DOMAIN:", target.domain)
    print("=" * 60)

    # ------------------------------------------------------
    # Forward search first.
    # ------------------------------------------------------

    clue = forward_search(target)

    # ------------------------------------------------------
    # If forward search fails, begin backward search.
    # ------------------------------------------------------

    if not clue:
        clue = backward_search(target)

    # ------------------------------------------------------
    # Final result.
    # ------------------------------------------------------

    print()
    print("=" * 60)

    if clue:
        print("FINAL GEOMETRY CLUE:", clue)
    else:
        print("NO GEOMETRY CLUE FOUND")
