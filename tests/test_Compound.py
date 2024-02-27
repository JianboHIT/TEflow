import unittest

from teflow.loader import Compound


class TestCompound(unittest.TestCase):
    def test_parse_1(self):
        comp = Compound.from_string('Mg3.2 Sb1.5 Bi0.48 Te0.02')
        ref = Compound([('Mg', 3.2), ('Sb', 1.5), ('Bi', 0.48), ('Te', 0.02)])
        self.assertEqual(comp, ref)
        self.assertEqual(Compound.from_string(comp.to_string(style='origin')), ref)
        self.assertAlmostEqual(comp.natom, 5.2)
        self.assertAlmostEqual(comp.weight_ave, 69.861, 2)

    def test_replace(self):
        comp = Compound.from_string('Mg3.2 Sb1.5 Bi0.48 Te0.02')
        ref = Compound.from_string('Mg3.2 As1.5 Bi0.48 Te0.02')
        comp.replace('Sb', 'As', match_order=True)
        self.assertEqual(comp, ref)
