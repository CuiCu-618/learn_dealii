/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */



#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>


#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>

using namespace dealii;

template <int dim>
class Solution : public Function<dim>
{
public:
    virtual double value(const Point<dim>& p,
                         const unsigned int component = 0) const override;
    virtual Tensor<1, dim> gradient(const Point<dim>& p,
                         const unsigned int component = 0) const override;
};

template <int dim>
double Solution<dim>::value(const Point<dim> &p, const unsigned int) const {
    return std::sin(3 * numbers::PI * p[0]) * std::sin(2 * numbers::PI * p[1]);
//    if (p.square() < 0.5 * 0.5)
//        return 1;
//    else
//        return 0;
}

template <int dim>
Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const {
    Tensor<1, dim> return_val;
    return_val[0] = 3 * numbers::PI * std::cos(3 * numbers::PI * p[0]) *
                                      std::sin(2 * numbers::PI * p[1]);
    return_val[1] = 2 * numbers::PI * std::sin(3 * numbers::PI * p[0]) *
                                      std::cos(2 * numbers::PI * p[1]);

//    return_val[0] = 0;
//    return_val[1] = 0;
    return return_val;
}

template <int dim>
class RightHandSide : public Function<dim>
{
public:
    virtual double value(const Point<dim>& p,
                         const unsigned int component = 0) const override;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
    return (13 * numbers::PI * numbers::PI + 0) *
                std::sin(3 * numbers::PI * p[0]) * std::sin(2 * numbers::PI * p[1]);
//    return -3 * std::cos(3 * numbers::PI * p[0]) * std::sin(2 * numbers::PI * p[1]) -
//            2 * std::sin(3 * numbers::PI * p[0]) * std::cos(2 * numbers::PI * p[1]);

}

template <int dim>
Tensor<1, dim> beta(const Point<dim> &p)
{
    Assert(dim >= 2, ExcNotImplemented());
    Tensor<1, dim> wind_field;
//    wind_field[0] = -p[1];
//    wind_field[1] = p[0];
//    if (wind_field.norm() > 1e-10)
//        wind_field /= wind_field.norm();
    wind_field[0] = 1;
    wind_field[1] = 1;
    return wind_field;
}

template <int dim>
class Step6
{
public:

  enum class RefinementMode {global, adaptive};

  Step6(FiniteElement<dim> & fe, const RefinementMode refinement_mode);

  void run();

private:
  void setup_system();
  void assemble_system();
  void solve();
  void refine_grid();
  void process_solution(const unsigned cycle);
  void output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation;

  // ? fe life time ?
//  const FE_Q<dim>       fe;
  SmartPointer<const FiniteElement<dim>> fe;
  DoFHandler<dim>       dof_handler;


  AffineConstraints<double> constraints;

  SparseMatrix<double> system_matrix;
  SparsityPattern      sparsity_pattern;

  Vector<double> solution;
  Vector<double> system_rhs;

  const RefinementMode refinement_mode;
  ConvergenceTable     convergence_table;
};



template <int dim>
double coefficient(const Point<dim> &p)
{
    return 1;

  if (p.square() < 0.5 * 0.5)
    return 20;
  else
    return 1;
}





template <int dim>
Step6<dim>::Step6(FiniteElement<dim> &fe, const RefinementMode refinement_mode)
  : fe(&fe)
  , dof_handler(triangulation)
  , refinement_mode(refinement_mode)
{}




template <int dim>
void Step6<dim>::setup_system()
{
  dof_handler.distribute_dofs(*fe);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);


  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Solution<dim>(),
                                           constraints);

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}



template <int dim>
void Step6<dim>::assemble_system()
{
  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss<dim - 1> face_quadrature_formula(fe->degree + 1);
  FEValues<dim> fe_values(*fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*fe,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                   update_normal_vectors | update_JxW_values);

  const unsigned int dofs_per_cell = fe->n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  Solution<dim> exact_solution;
  const RightHandSide<dim> right_hand_side;
  std::vector<double>      rhs_values(quadrature_formula.size());

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);

      right_hand_side.value_list(fe_values.get_quadrature_points(),
                                 rhs_values);

      const auto &q_points = fe_values.get_quadrature_points();

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          const double current_coefficient =
            coefficient(q_points[q_index]);

          auto beta_q = beta(q_points[q_index]);
          for (const unsigned int i : fe_values.dof_indices())
            {
              for (const unsigned int j : fe_values.dof_indices())
                cell_matrix(i, j) +=
                  ((current_coefficient *              // a(x_q)
                   fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                   fe_values.shape_grad(j, q_index)   // grad phi_j(x_q)
                   +
                   fe_values.shape_value(i, q_index) *
                   fe_values.shape_value(j, q_index) * 0
                   +
                   beta_q * 0 *
                   fe_values.shape_value(i, q_index) *
                   fe_values.shape_grad(j, q_index)) *
                   fe_values.JxW(q_index));

              cell_rhs(i) += (rhs_values[q_index] *               // f(x)
                              fe_values.shape_value(i, q_index) * // phi_i(x_q)
                              fe_values.JxW(q_index));            // dx
            }
        }

      // Neumann B.C.
      for (auto &face : cell->face_iterators())
          if (face->at_boundary() && face->boundary_id() == 1)
          {
              fe_face_values.reinit(cell, face);

              for (const unsigned q_index : fe_face_values.quadrature_point_indices()){
                  const double neumann_value =
                          exact_solution.gradient(fe_face_values.quadrature_point(q_index)) *
                          fe_face_values.normal_vector(q_index);

                  const double boundary_value =
                          exact_solution.value(fe_face_values.quadrature_point(q_index));

                  const double beta_dot_n = beta(fe_face_values.quadrature_point(q_index)) *
                          fe_face_values.normal_vector(q_index);

                  for (const unsigned int i : fe_face_values.dof_indices()){
                      cell_rhs(i) += ((fe_face_values.shape_value(i, q_index) *
                                     neumann_value
                                     +
                                     beta_dot_n * 0 *
                                     fe_face_values.shape_value(i, q_index) *
                                     boundary_value) *
                                     fe_face_values.JxW(q_index));
                  }
              }
          }


      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}




template <int dim>
void Step6<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
//  SolverGMRES<Vector<double>> solver(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);
}



template <int dim>
void Step6<dim>::refine_grid()
{
    switch (refinement_mode) {
        case RefinementMode::global:
        {
            triangulation.refine_global(1);
            break;
        }
        case RefinementMode::adaptive:
        {
            Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

            KellyErrorEstimator<dim>::estimate(dof_handler,
                                               QGauss<dim - 1>(fe->degree + 1),
                                               std::map<types::boundary_id, const Function<dim> *>(),
                                               solution,
                                               estimated_error_per_cell);

            GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                            estimated_error_per_cell,
                                                            0.3,
                                                            0.03);
            break;
        }
        default:
        {
            Assert(false, ExcNotImplemented());
        }

    }
  triangulation.execute_coarsening_and_refinement();
}



template <int dim>
void Step6<dim>::output_results(const unsigned int cycle) const
{

  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();

    std::string filename = "solution";
      switch (refinement_mode) {
          case RefinementMode::global:
              filename += "-global";
              break;
          case RefinementMode::adaptive:
              filename += "-adaptive";
              break;
          default: Assert(false, ExcNotImplemented());
      }
    filename += std::to_string(cycle);
    filename += ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }
}

template <int dim>
void Step6<dim>::process_solution(const unsigned int cycle)
{
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(fe->degree + 1),
                                      VectorTools::L2_norm);
    const double L2_error =
            VectorTools::compute_global_error(triangulation,
                                              difference_per_cell,
                                              VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(fe->degree + 1),
                                      VectorTools::H1_seminorm);
    const double H1_error =
            VectorTools::compute_global_error(triangulation,
                                              difference_per_cell,
                                              VectorTools::H1_seminorm);
    const QTrapezoid<1>  q_trapez;
    const QIterated<dim> q_iterated(q_trapez, fe->degree * 2 + 1);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      q_iterated,
                                      VectorTools::Linfty_norm);
    const double Linfty_error =
            VectorTools::compute_global_error(triangulation,
                                              difference_per_cell,
                                              VectorTools::Linfty_norm);
    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs         = dof_handler.n_dofs();
    std::cout << "Cycle " << cycle << ':' << std::endl
    << "   Number of active cells:       " << n_active_cells
    << std::endl
    << "   Number of degrees of freedom: " << n_dofs << std::endl;
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("Linfty", Linfty_error);
}

template <int dim>
void Step6<dim>::run()
{
    const unsigned int n_cycles =
            (refinement_mode == RefinementMode::global) ? 4 : 8;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {

      if (cycle == 0)
        {
          GridGenerator::hyper_cube(triangulation);
          triangulation.begin_active()->face(0)->set_boundary_id(1);
          triangulation.begin_active()->face(1)->set_boundary_id(1);

          triangulation.refine_global(2);
        }
      else
        refine_grid();

      setup_system();
      assemble_system();
      solve();
      output_results(cycle);
      process_solution(cycle);
    }

  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("H1", 3);
  convergence_table.set_precision("Linfty", 3);
  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);
  convergence_table.set_scientific("Linfty", true);

  std::cout << std::endl;
  convergence_table.write_text(std::cout);

  if (refinement_mode == RefinementMode::global)
  {
    convergence_table.add_column_to_supercolumn("cycle", "n cells");
    convergence_table.add_column_to_supercolumn("cells", "n cells");
    std::vector<std::string> new_order;
    new_order.emplace_back("n cells");
    new_order.emplace_back("H1");
    new_order.emplace_back("L2");
    convergence_table.set_column_order(new_order);
    convergence_table.evaluate_convergence_rates(
            "L2", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
            "L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
            "H1", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
            "H1", ConvergenceTable::reduction_rate_log2);
    std::cout << std::endl;
    convergence_table.write_text(std::cout);
  }
}



int main()
{
  try
    {
      using namespace dealii;
      {
          std::cout << "Solving with Q1 elements, global refinement"
          << std::endl
          << "============================================="
          << std::endl
          << std::endl;

          FE_Q<2> fe(2);

          Step6<2> laplace_problem_2d(fe, Step6<2>::RefinementMode::global);
          laplace_problem_2d.run();
          std::cout << std::endl;
      }

      {
          std::cout << "Solving with Q1 elements, adaptive refinement"
          << std::endl
          << "============================================="
          << std::endl
          << std::endl;

          FE_Q<2> fe(2);

          Step6<2> laplace_problem_2d(fe, Step6<2>::RefinementMode::adaptive);
          laplace_problem_2d.run();
          std::cout << std::endl;
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
