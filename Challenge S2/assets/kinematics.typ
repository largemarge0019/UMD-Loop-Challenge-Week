#import "@preview/cetz:0.5.2"

= Inverse Kinematics of the Main Arm

We start with the diagrams of the arm, split into a top view and a side view
#columns(2)[
  #figure(
    cetz.canvas({
      import cetz.draw: *
      import cetz.angle: angle

      line((-2.5, -2.5), (2.5, -2.5), mark: (end: ">"))
      content((2.8, -2.5), [$x$])
      line((-2.5, -2.5), (-2.5, 2.5), mark: (end: ">"))
      content((-2.5, 2.8), [$y$])

      circle((0, 0), radius: 2)
      circle((0, 0), radius: 0.08, fill: black, stroke: none)
      line((0, 0), (angle: 32deg, radius: 2))
      circle((angle: 32deg, radius: 2), radius: 0.08, fill: black, stroke: none)
      line((0, 0), (2, 0), stroke: (dash: "dashed"))
      content((angle: 32deg, radius: 2.3), [$arrow(r)$])
      angle((0, 0), (1, 0), (angle: 32deg, radius: 1), label: [$theta_0$], radius: 1, label-radius: 1.3)
    }),
    caption: "Top View",
  )

  #colbreak()

  #figure(
    cetz.canvas({
      import cetz.draw: *
      import cetz.angle: angle

      let upper-len = 2.4
      let fore-len = 1.6
      let upper-angle = 85deg
      let relative-angle = -113deg
      let label-offset = 0.3

      let point(pos, label: none, offset: (0.3, 0.3)) = {
        circle(pos, radius: 0.08, fill: black, stroke: none)

        if label != none {
          content(
            (
              pos.at(0) + offset.at(0),
              pos.at(1) + offset.at(1),
            ),
            label,
            anchor: "center",
          )
        }
      }

      let shoulder = (0, -2)
      let elbow = (
        shoulder.at(0) + upper-len * calc.cos(upper-angle),
        shoulder.at(1) + upper-len * calc.sin(upper-angle),
      )

      let fore-angle = upper-angle + relative-angle
      let wrist = (
        elbow.at(0) + fore-len * calc.cos(fore-angle),
        elbow.at(1) + fore-len * calc.sin(fore-angle),
      )

      line((-1.5, -2.5), (3.5, -2.5), mark: (end: ">"))
      content((3.8, -2.5), [$r$])
      line((-1.5, -2.5), (-1.5, 2.5), mark: (end: ">"))
      content((-1.5, 2.8), [$z$])

      line((-1, -2), (2, -2), stroke: (dash: "dashed"))

      line(shoulder, elbow)
      line(elbow, wrist)

      point(shoulder, label: [$arrow(p_0)$], offset: (0, -0.3))
      point(elbow, label: [$arrow(p_1)$], offset: (-0.3, 0.3))
      point(wrist, label: [$arrow(p_2)$])

      angle(
        shoulder,
        (2, -2),
        elbow,
        label: [$theta_1$],
        label-radius: 0.75,
      )

      let elbow-extended = (
        (elbow.at(0) - shoulder.at(0)) * 1.3 + shoulder.at(0),
        (elbow.at(1) - shoulder.at(1)) * 1.3 + shoulder.at(1),
      )

      line(elbow, elbow-extended, stroke: (dash: "dashed"))
      angle(
        elbow,
        elbow-extended,
        wrist,
        direction: "cw",
        label: [$theta_2$],
        label-radius: 0.8,
      )

      let upper-mid = (
        (shoulder.at(0) + elbow.at(0)) / 2,
        (shoulder.at(1) + elbow.at(1)) / 2,
      )

      let upper-label = (
        upper-mid.at(0) - label-offset * calc.sin(upper-angle),
        upper-mid.at(1) + label-offset * calc.cos(upper-angle),
      )

      let fore-mid = (
        (elbow.at(0) + wrist.at(0)) / 2,
        (elbow.at(1) + wrist.at(1)) / 2,
      )

      let fore-label = (
        fore-mid.at(0) - label-offset * calc.sin(fore-angle),
        fore-mid.at(1) + label-offset * calc.cos(fore-angle),
      )

      content(upper-label, [$L_1$])
      content(fore-label, [$L_2$])

      line(wrist, shoulder, stroke: (dash: "dashed"))

      let dashed-mid = (
        (wrist.at(0) + shoulder.at(0)) / 2,
        (wrist.at(1) + shoulder.at(1)) / 2,
      )

      let dashed-angle = calc.atan2(
        wrist.at(1) - shoulder.at(1),
        wrist.at(0) - shoulder.at(0),
      )

      let dashed-label = (
        dashed-mid.at(0) + label-offset * calc.sin(dashed-angle),
        dashed-mid.at(1) - label-offset * calc.cos(dashed-angle),
      )

      content(dashed-label, [$d$])
    }),
    caption: "Side View",
  )
]

We already know $L_1$ and $L_2$, as they are the lengths of the robot's arms which are given. $arrow(p_0)$ and $arrow(p_2)$ are also known as they are the position of the base which is static and the goal position of the end effector.

We are trying to solve for $theta_0$, $theta_1$ and $theta_2$, which are the joint angles the robot needs to reach the desired positions.

To solve this first notice that $arrow(p_0)$, $arrow(p_1)$, and $arrow(p_2)$ make a triangle. We also know all 3 side lengths so we can use the law of cosines to figure out each of the interior angles and then convert them into the angles we need with some more simple math.

*Definition: * The Law of Cosines - $c^2 = a^2 + b^2 - 2 a b cos(C)$ where $C$ is the angle opposite side $c$

$ d = |arrow(p_2) - arrow(p_0)| $

Solve for $theta_2$

$
                          d^2 & = L_1^2 + L_2^2 - 2 L_1 L_2 cos(pi - theta_2) \
  2 L_1 L_2 cos(pi - theta_2) & = L_1^2 + L_2^2 - d^2 \
            cos(pi - theta_2) & = (L_1^2 + L_2 - d^2)/(2 L_1 L_2) \
                -cos(theta_2) & = (L_1^2 + L_2 - d^2)/(2 L_1 L_2) \
                      theta_2 & = cos^(-1)((d^2 - L_1^2 - L^2^2) / (2 L_1 L_2))
$

Solve for $theta_1$ \
First we solve for the interior angle of the triangle, opposite to $L_2$, the same way as we did for $theta_2$
then to get the angle from the $r$-axis to $arrow(p_2)$, we can do $"atan2"(arrow(p_2))$. Therefore
$ theta_1 = "atan2"(arrow(p_2)) + cos^(-1)((L_1^2 + d^2 - L_2^2) / (2 L_1 d)) $

#pagebreak()

And finally, to solve for $theta_0$ we can just do
$ theta_0 = "atan2"(arrow(r)) $

To put it all together, to find the joint angles needed to get the end effector to $arrow(p_2)$, we have the following equations
$
  theta_0 & = "atan2"(arrow(p_2)_y, arrow(p_2)_x) \
  theta_1 & = "atan2"(arrow(p_2)_z, sqrt(arrow(p_2)_x^2 + arrow(p_2)_y^2)) + cos^(-1)((L_1^2 + |arrow(p_2) - arrow(p_0)|^2 - L_2^2)/(2 L_1 |arrow(p_2) - arrow(p_0)|)) \
  theta_2 & = cos^(-1)((|arrow(p_2) - arrow(p_0)|^2 - L_1^2 - L_2^2) / (2 L_1 L_2))
$
