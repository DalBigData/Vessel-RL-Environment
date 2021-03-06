"""
  \ file circle_2d.py
  \ brief 2D circle region File.
"""

from server.math.triangle_2d import *
from server.math.segment_2d import *

"""
  \ brief solve quadratic formula
  \ param a formula constant A
  \ param b formula constant B
  \ param c formula constant C
  \ param sol1 reference to the result variable
  \ param sol2 reference to the result variable
  \ return number of solution
 """


def QUADRATIC_F(a, b, c):
    d = b * b - 4.0 * a * c
    sol1 = 0.0
    sol2 = 0.0
    if math.fabs(d) < EPSILON:
        sol1 = -b / (2.0 * a)
        ans = 1
    elif d < 0.0:
        ans = 0
    else:
        d = math.sqrt(d)
        sol1 = (-b + d) / (2.0 * a)
        sol2 = (-b - d) / (2.0 * a)
        ans = 2
    return [ans, sol1, sol2]


def SQUARE(v):
    return v * v


class Circle2D:
    """
        Default:
      \ brief create a zero area circle at (0,0)
        Len = 2
      \ brief construct with center point and radius value.
      \ param c center point
      \ param r radius value
    """

    def __init__(self, *args):  # , **kwargs):)
        if len(args) == 2 and isinstance(args[0], Vector2D):
            self._center = args[0]
            self._radius = args[1]
            if args[1] < 0.0:
                self._radius = 0.0
            self._is_valid = True
        else:
            self._center = Vector2D()
            self._radius = 0.0
            self._is_valid = True

    """
      \ brief assign value.
      \ param c center point
      \ param r radius value
    """

    def assign(self, c: Vector2D, r: float):
        self._center = c
        self._radius = r
        if r < 0.0:
            self._radius = 0.0

    """
      \ brief get the area value of self circle
      \ return value of the area
     """

    def area(self):
        return PI * self._radius * self._radius

    """
      \ brief check if point is within self region
      \ param point considered point
      \ return True if point is contained by self circle
     """

    def contains(self, point: Vector2D):
        return self._center.dist2(point) < self._radius * self._radius

    """
      \ brief get the center point
      \ return center point coordinate value
     """

    def center(self):
        return self._center

    """
      \ brief get the radius value
      \ return radius value
    """

    def radius(self):
        return self._radius

    """
        Line2D
      \ brief calculate the intersection with straight line
      \ param line considered line
      \ return the number of solution + solutions
        Ray2D
      \ brief calculate the intersection with ray line
      \ param ray considered ray
      \ return the number of solution + solutions
        Segment2D
      \ brief calculate the intersection with segment line
      \ param segment considered segment line
      \ return the number of solution + solutions
        Circle2D
      \ brief calculate the intersection with another circle
      \ param circle considered circle
      \ return the number of solution + solutions
    """

    def intersection(self, *args):  # , **kwargs):):):
        if len(args) == 1 and isinstance(args[0], Line2D):
            line = args[0]
            if math.fabs(line.a()) < EPSILON:
                if math.fabs(line.b()) < EPSILON:
                    return 0

                n_sol = QUADRATIC_F(1.0,
                                    -2.0 * self._center._x,
                                    (SQUARE(self._center._x)
                                     + SQUARE(line.c() / line.b() + self._center._y)
                                     - SQUARE(self._radius)))
                x1 = n_sol[1]
                x2 = n_sol[2]
                if n_sol[0] > 0:
                    y1 = -line.c() / line.b()
                    sol_list = [n_sol[0], Vector2D(x1, y1), Vector2D(x2, y1)]
                else:
                    sol_list = [n_sol[0]]
                return sol_list

            else:
                m = line.b() / line.a()
                d = line.c() / line.a()

                a = 1.0 + m * m
                b = 2.0 * (-self._center._y + (d + self._center._x) * m)
                c = SQUARE(d + self._center._x) + SQUARE(self._center._y) - SQUARE(self._radius)

            n_sol = QUADRATIC_F(a, b, c)
            y1 = n_sol[1]
            y2 = n_sol[2]
            sol_list = [n_sol[0], Vector2D(line.getX(y1), y1), Vector2D(line.getX(y2), y2)]
            return sol_list
        elif len(args) == 1 and isinstance(args[0], Ray2D):
            ray = args[0]
            line_tmp = Line2D(ray.origin(), ray.dir())

            n_sol = self.intersection(line_tmp)
            t_sol1 = n_sol[1]
            t_sol2 = n_sol[2]

            if n_sol[0] > 1 and not ray.inRightDir(t_sol2, 1.0):
                n_sol[0] -= 1

            if n_sol[0] > 0 and not ray.inRightDir(t_sol1, 1.0):
                t_sol1 = t_sol2
                n_sol[0] -= 1

            sol_list = [n_sol[0], t_sol1, t_sol2]

            return sol_list

        elif len(args) == 1 and isinstance(args[0], Segment2D):
            seg = args[0]
            line = seg.line()
            t_sol1 = Vector2D()
            t_sol2 = Vector2D()

            n_sol = self.intersection(line)

            if n_sol[0] > 1 and not seg.contains(t_sol2):
                n_sol[0] -= 1

            if n_sol[0] > 0 and not seg.contains(t_sol1):
                n_sol[0] -= 1

            sol_list = [n_sol[0], t_sol1, t_sol2]

            return sol_list

        elif len(args) == 1 and isinstance(args[0], Circle2D):
            circle = args[0]

            rel_x = circle.center().x() - self._center._x
            rel_y = circle.center().y() - self._center._y

            center_dist2 = rel_x * rel_x + rel_y * rel_y
            center_dist = math.sqrt(center_dist2)

            if center_dist < math.fabs(self._radius - circle.radius()) or self._radius + circle.radius() < center_dist:
                return

            line = Line2D(-2.0 * rel_x, -2.0 * rel_y,
                          circle.center().r2() - circle.radius() * circle.radius() - self._center.r2() + self._radius * self._radius)

            return self.intersection(line)

    def has_intersection(self, *args):
        if isinstance(args[0], Circle2D):
            circle = args[0]

            rel_x = circle.center().x() - self._center._x
            rel_y = circle.center().y() - self._center._y

            center_dist2 = rel_x * rel_x + rel_y * rel_y
            center_dist = math.sqrt(center_dist2)

            if self._radius + circle.radius() < center_dist:
                return False
            return True
        elif isinstance(args[0], Segment2D):
            seg = args[0]
            line = seg.line()
            t_sol1 = Vector2D()
            t_sol2 = Vector2D()

            n_sol = self.intersection(line)
            print(n_sol)
            if n_sol[0] > 1 and not seg.contains(t_sol2):
                print('not 2')
                n_sol[0] -= 1

            if n_sol[0] > 0 and not seg.contains(t_sol1):
                print('not 1')
                n_sol[0] -= 1

            if n_sol[0] > 0:
                return True
            return False

    """  ----------------- static method  ----------------- """

    """
      \ brief get the circle through three points (circumcircle of the triangle).
      \ param p0 triangle's 1st vertex
      \ param p1 triangle's 2nd vertex
      \ param p2 triangle's 3rd vertex
      \ return coordinates of circumcenter
    """

    @staticmethod
    def circumcircle(p0, p1, p2):
        center = Triangle2D.tri_circumcenter(p0, p1, p2)

        if not center.is_valid():
            return Circle2D()

        return Circle2D(center, center.dist(p0))

    """
      \ brief check if the circumcircle contains the input point
      \ param point input point
      \ param p0 triangle's 1st vertex
      \ param p1 triangle's 2nd vertex
      \ param p2 triangle's 3rd vertex
      \ return True if circumcircle contains the point, False.
    """

    @staticmethod
    def circle_contains(point, p0, p1, p2):
        a = p1.x - p0.x
        b = p1.y - p0.y
        c = p2.x - p0.x
        d = p2.y - p0.y

        e = a * (p0.x + p1.x) + b * (p0.y + p1.y)
        f = c * (p0.x + p2.x) + d * (p0.y + p2.y)

        g = 2.0 * (a * (p2.y - p1.y) - b * (p2.x - p1.x))
        if math.fabs(g) < 1.0e-10:
            return False

        center = Vector2D((d * e - b * f) / g, (a * f - c * e) / g)
        return center.dist2(point) < center.dist2(p0) - EPSILON * EPSILON

    """
      \ brief make a logical print.
      \ return print_able str
    """

    def __repr__(self):
        return "({} , {})".format(self._center, self._radius)


def test():
    c = Circle2D()
    print(c)


if __name__ == "__main__":
    test()
