"""Stack red on green and blue on red."""

INSTRUCTION = "Place red on green with the left arm, then blue on red with the right arm."


def stack_from_real_layout(skills):
    left = skills.pick("red", arm="left")
    skills.place_on("red", "green", left)
    skills.park(left)
    right = skills.pick("blue", arm="right")
    skills.place_on("blue", "red", right)
    skills.park(right)
    return skills.finish()
