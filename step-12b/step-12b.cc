/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2021 by the deal.II authors
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
 * Author: Guido Kanschat, Texas A&M University, 2009
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>


namespace Step12
{
  using namespace dealii;

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
      virtual double value(const Point <dim> &p,
                           const unsigned int component = 0) const override;

      virtual void value_list(const std::vector<Point<dim>> &points,
                              std::vector<double> &          values,
                              const unsigned int component = 0) const override;

      virtual Tensor<1, dim> gradient(const Point <dim> &p,
                                      const unsigned int component = 0) const override;
  };

  template <int dim>
  double Solution<dim>::value(const Point<dim> &p, const unsigned int) const {
      return std::cos(numbers::PI * p[0] * p[1]);
  }

  template <int dim>
  Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const
  {
      Tensor<1, dim> return_val;
      return_val[0] = -p[1] * numbers::PI * std::sin(numbers::PI * p[0] * p[1]);
      return_val[1] = -p[0] * numbers::PI * std::sin(numbers::PI * p[0] * p[1]);

      return return_val;
  }

  template<int dim>
  void Solution<dim>::value_list(const std::vector<Point<dim>> &points,
                                 std::vector<double> &          values,
                                 const unsigned int component) const
  {
      (void)component;
      AssertIndexRange(component, 1);
      Assert(values.size() == points.size(),
             ExcDimensionMismatch(values.size(), points.size()));

      for (unsigned int i = 0; i < values.size(); ++i){
          values[i] = this->value(points[i]);
      }
  }

  template <int dim>
  class RightHandSide : public Function<dim>
          {
          public:
              virtual double value(const Point<dim>& p,
                                   const unsigned int component = 0) const override;
              virtual void value_list(const std::vector<Point<dim>> &points,
                                      std::vector<double> &          values,
                                      const unsigned int component = 0) const override;
          };

  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
      return - (p[1] + p[0]) * numbers::PI * std::sin(numbers::PI * p[0] * p[1]);
  }

  template<int dim>
  void RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                                      std::vector<double> &          values,
                                      const unsigned int component) const
  {
      (void)component;
      AssertIndexRange(component, 1);
      Assert(values.size() == points.size(),
             ExcDimensionMismatch(values.size(), points.size()));

      for (unsigned int i = 0; i < values.size(); ++i){
          values[i] = this->value(points[i]);
      }
  }

  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() = default;
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double> &          values,
                            const unsigned int component = 0) const override;
  };

  template <int dim>
  void BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                       std::vector<double> &          values,
                                       const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        if (points[i](0) < 0.5)
          values[i] = 1.;
        else
          values[i] = 0.;
      }
  }


  template <int dim>
  Tensor<1, dim> beta(const Point<dim> &p)
  {
    Assert(dim >= 2, ExcNotImplemented());

    Tensor<1, dim> wind_field;

    wind_field[0] = 1;
    wind_field[1] = 1;

//    wind_field[0] = -p[0];
//    wind_field[1] = p[1];
//    wind_field /= wind_field.norm();

    return wind_field;
  }



  template <int dim>
  class AdvectionProblem
  {
  public:
    enum RefinementMode {global, adaptive};

    AdvectionProblem(const RefinementMode refinement_mode);
    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve(Vector<double> &solution);
    void refine_grid();
    void process_solution(const unsigned cycle);
    void output_results(const unsigned int cycle) const;

    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;

    FE_DGQ<dim>     fe;
    DoFHandler<dim> dof_handler;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> right_hand_side;

    ConvergenceTable     convergence_table;
    const RefinementMode refinement_mode;

    using DoFInfo  = MeshWorker::DoFInfo<dim>;
    using CellInfo = MeshWorker::IntegrationInfo<dim>;

    static void integrate_cell_term(DoFInfo &dinfo, CellInfo &info);
    static void integrate_boundary_term(DoFInfo &dinfo, CellInfo &info);
    static void integrate_face_term(DoFInfo & dinfo1,
                                    DoFInfo & dinfo2,
                                    CellInfo &info1,
                                    CellInfo &info2);
  };


  template <int dim>
  AdvectionProblem<dim>::AdvectionProblem(const RefinementMode refinement_mode)
    : mapping()
    , fe(2)
    , dof_handler(triangulation)
    , refinement_mode(refinement_mode)
  {}


  template <int dim>
  void AdvectionProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);


    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());
  }


  template <int dim>
  void AdvectionProblem<dim>::assemble_system()
  {
    MeshWorker::IntegrationInfoBox<dim> info_box;

    const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;
    info_box.initialize_gauss_quadrature(n_gauss_points,
                                         n_gauss_points,
                                         n_gauss_points);

    info_box.initialize_update_flags();
    UpdateFlags update_flags =
      update_quadrature_points | update_values | update_gradients;
    info_box.add_update_flags(update_flags, true, true, true, true);

    info_box.initialize(fe, mapping);

    MeshWorker::DoFInfo<dim> dof_info(dof_handler);

    MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>>
      assembler;
    assembler.initialize(system_matrix, right_hand_side);

    MeshWorker::loop<dim,
                     dim,
                     MeshWorker::DoFInfo<dim>,
                     MeshWorker::IntegrationInfoBox<dim>>(
      dof_handler.begin_active(),
      dof_handler.end(),
      dof_info,
      info_box,
      &AdvectionProblem<dim>::integrate_cell_term,
      &AdvectionProblem<dim>::integrate_boundary_term,
      &AdvectionProblem<dim>::integrate_face_term,
      assembler);
  }



  template <int dim>
  void AdvectionProblem<dim>::integrate_cell_term(DoFInfo & dinfo,
                                                  CellInfo &info)
  {
    const FEValuesBase<dim> &  fe_values    = info.fe_values();
    FullMatrix<double> &       local_matrix = dinfo.matrix(0).matrix;
    Vector<double> &           local_vector = dinfo.vector(0).block(0);
    const std::vector<double> &JxW          = fe_values.get_JxW_values();

    const RightHandSide<dim> right_hand_side;
    std::vector<double>      rhs_values(fe_values.n_quadrature_points);
    right_hand_side.value_list(fe_values.get_quadrature_points(),
                               rhs_values);

    for (unsigned int point = 0; point < fe_values.n_quadrature_points; ++point)
      {
        const Tensor<1, dim> beta_at_q_point =
          beta(fe_values.quadrature_point(point));

        for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
        {
            for (unsigned int j = 0; j < fe_values.dofs_per_cell; ++j)
            {
                local_matrix(i, j) += -beta_at_q_point *                //
                        fe_values.shape_grad(i, point) *  //
                        fe_values.shape_value(j, point) * //
                        JxW[point];

            }

            local_vector(i) += rhs_values[point] *               // f(x)
                    fe_values.shape_value(i, point) * // phi_i(x_q)
                    fe_values.JxW(point);

        }
      }
  }

  template <int dim>
  void AdvectionProblem<dim>::integrate_boundary_term(DoFInfo & dinfo,
                                                      CellInfo &info)
  {
    const FEValuesBase<dim> &fe_face_values = info.fe_values();
    FullMatrix<double> &     local_matrix   = dinfo.matrix(0).matrix;
    Vector<double> &         local_vector   = dinfo.vector(0).block(0);

    const std::vector<double> &        JxW = fe_face_values.get_JxW_values();
    const std::vector<Tensor<1, dim>> &normals =
      fe_face_values.get_normal_vectors();

    std::vector<double> g(fe_face_values.n_quadrature_points);

//    static BoundaryValues<dim> boundary_function;
    const Solution<dim> boundary_function;
    boundary_function.value_list(fe_face_values.get_quadrature_points(), g);

    for (unsigned int point = 0; point < fe_face_values.n_quadrature_points;
         ++point)
      {
        const double beta_dot_n =
          beta(fe_face_values.quadrature_point(point)) * normals[point];
        if (beta_dot_n > 0)
          for (unsigned int i = 0; i < fe_face_values.dofs_per_cell; ++i)
            for (unsigned int j = 0; j < fe_face_values.dofs_per_cell; ++j)
              local_matrix(i, j) += beta_dot_n *                           //
                                    fe_face_values.shape_value(j, point) * //
                                    fe_face_values.shape_value(i, point) * //
                                    JxW[point];
        else
          for (unsigned int i = 0; i < fe_face_values.dofs_per_cell; ++i)
            local_vector(i) += -beta_dot_n *                          //
                               g[point] *                             //
                               fe_face_values.shape_value(i, point) * //
                               JxW[point];
      }
  }

  template <int dim>
  void AdvectionProblem<dim>::integrate_face_term(DoFInfo & dinfo1,
                                                  DoFInfo & dinfo2,
                                                  CellInfo &info1,
                                                  CellInfo &info2)
  {
    const FEValuesBase<dim> &fe_face_values = info1.fe_values();
    const unsigned int       dofs_per_cell  = fe_face_values.dofs_per_cell;

    const FEValuesBase<dim> &fe_face_values_neighbor = info2.fe_values();
    const unsigned int       neighbor_dofs_per_cell =
      fe_face_values_neighbor.dofs_per_cell;

    FullMatrix<double> &u1_v1_matrix = dinfo1.matrix(0, false).matrix;
    FullMatrix<double> &u2_v1_matrix = dinfo1.matrix(0, true).matrix;
    FullMatrix<double> &u1_v2_matrix = dinfo2.matrix(0, true).matrix;
    FullMatrix<double> &u2_v2_matrix = dinfo2.matrix(0, false).matrix;


    const std::vector<double> &        JxW = fe_face_values.get_JxW_values();
    const std::vector<Tensor<1, dim>> &normals =
      fe_face_values.get_normal_vectors();

    for (unsigned int point = 0; point < fe_face_values.n_quadrature_points;
         ++point)
      {
        const double beta_dot_n =
          beta(fe_face_values.quadrature_point(point)) * normals[point];
        if (beta_dot_n > 0)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                u1_v1_matrix(i, j) += beta_dot_n *                           //
                                      fe_face_values.shape_value(j, point) * //
                                      fe_face_values.shape_value(i, point) * //
                                      JxW[point];

            for (unsigned int k = 0; k < neighbor_dofs_per_cell; ++k)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                u1_v2_matrix(k, j) +=
                  -beta_dot_n *                                   //
                  fe_face_values.shape_value(j, point) *          //
                  fe_face_values_neighbor.shape_value(k, point) * //
                  JxW[point];
          }
        else
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int l = 0; l < neighbor_dofs_per_cell; ++l)
                u2_v1_matrix(i, l) +=
                  beta_dot_n *                                    //
                  fe_face_values_neighbor.shape_value(l, point) * //
                  fe_face_values.shape_value(i, point) *          //
                  JxW[point];

            for (unsigned int k = 0; k < neighbor_dofs_per_cell; ++k)
              for (unsigned int l = 0; l < neighbor_dofs_per_cell; ++l)
                u2_v2_matrix(k, l) +=
                  -beta_dot_n *                                   //
                  fe_face_values_neighbor.shape_value(l, point) * //
                  fe_face_values_neighbor.shape_value(k, point) * //
                  JxW[point];
          }
      }
  }


  template <int dim>
  void AdvectionProblem<dim>::solve(Vector<double> &solution)
  {
    SolverControl                    solver_control(1000, 1e-12);
    SolverRichardson<Vector<double>> solver(solver_control);

    PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;

    preconditioner.initialize(system_matrix, fe.n_dofs_per_cell());

    solver.solve(system_matrix, solution, right_hand_side, preconditioner);
  }


  template <int dim>
  void AdvectionProblem<dim>::refine_grid()
  {

      switch (refinement_mode) {
          case global:
          {
              triangulation.refine_global(1);
              break;
          }
          case adaptive:
          {
              Vector<float> gradient_indicator(triangulation.n_active_cells());

              DerivativeApproximation::approximate_gradient(mapping,
                                                            dof_handler,
                                                            solution,
                                                            gradient_indicator);

              unsigned int cell_no = 0;
              for (const auto &cell : dof_handler.active_cell_iterators())
                  gradient_indicator(cell_no++) *=
                          std::pow(cell->diameter(), 1 + 1.0 * dim / 2);

              GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                              gradient_indicator,
                                                              0.3,
                                                              0.1);

              triangulation.execute_coarsening_and_refinement();
              break;
          }
          default:
          {
              Assert(false, ExcNotImplemented());
              break;
          }
      }

  }


  template <int dim>
  void AdvectionProblem<dim>::process_solution(const unsigned int cycle)
  {
      Vector<float> difference_per_cell(triangulation.n_active_cells());
      VectorTools::integrate_difference(dof_handler,
                                        solution,
                                        Solution<dim>(),
                                        difference_per_cell,
                                        QGauss<dim>(fe.degree + 1),
                                        VectorTools::L2_norm);
      const double L2_error =
              VectorTools::compute_global_error(triangulation,
                                                difference_per_cell,
                                                VectorTools::L2_norm);
      VectorTools::integrate_difference(dof_handler,
                                        solution,
                                        Solution<dim>(),
                                        difference_per_cell,
                                        QGauss<dim>(fe.degree + 1),
                                        VectorTools::H1_seminorm);
      const double H1_error =
              VectorTools::compute_global_error(triangulation,
                                                difference_per_cell,
                                                VectorTools::H1_seminorm);
      const QTrapezoid<1>  q_trapez;
      const QIterated<dim> q_iterated(q_trapez, fe.degree * 2 + 1);
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
  void AdvectionProblem<dim>::output_results(const unsigned int cycle) const
  {
    {
      const std::string filename = "grid-" + std::to_string(cycle) + ".eps";
      deallog << "Writing grid to <" << filename << ">" << std::endl;
      std::ofstream eps_output(filename);

      GridOut grid_out;
      grid_out.write_eps(triangulation, eps_output);
    }

    {
      const std::string filename = "sol-" + std::to_string(cycle) + ".vtk";
      deallog << "Writing solution to <" << filename << ">" << std::endl;
      std::ofstream vtk_output(filename);

      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "u");

      data_out.build_patches();

      data_out.write_vtk(vtk_output);
    }
  }


  template <int dim>
  void AdvectionProblem<dim>::run()
  {
    const unsigned int n_cycles =
              (refinement_mode == global) ? 5 : 8;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        deallog << "Cycle " << cycle << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation);

            triangulation.refine_global(2);
          }
        else
          refine_grid();


        deallog << "Number of active cells:       "
                << triangulation.n_active_cells() << std::endl;

        setup_system();

        deallog << "Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;

        assemble_system();
        solve(solution);
        process_solution(cycle);
        output_results(cycle);
      }

    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1", 3);
    convergence_table.set_precision("Linfty", 3);
    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("Linfty", true);

    std::cout << std::endl;
    convergence_table.write_text(std::cout);


      convergence_table.add_column_to_supercolumn("cycle", "n cells");
      convergence_table.add_column_to_supercolumn("cells", "n cells");
      std::vector <std::string> new_order;
      new_order.emplace_back("n cells");
      new_order.emplace_back("H1");
      new_order.emplace_back("L2");
      new_order.emplace_back("Linfty");
      convergence_table.set_column_order(new_order);
      convergence_table.evaluate_convergence_rates(
              "L2", ConvergenceTable::reduction_rate);
      convergence_table.evaluate_convergence_rates(
              "L2", ConvergenceTable::reduction_rate_log2);
      convergence_table.evaluate_convergence_rates(
              "H1", ConvergenceTable::reduction_rate);
      convergence_table.evaluate_convergence_rates(
              "H1", ConvergenceTable::reduction_rate_log2);
      convergence_table.evaluate_convergence_rates(
              "Linfty", ConvergenceTable::reduction_rate);
      convergence_table.evaluate_convergence_rates(
              "Linfty", ConvergenceTable::reduction_rate_log2);
      std::cout << std::endl;
      convergence_table.write_text(std::cout);

  }
} // namespace Step12


int main()
{
  try
  {
      //      dealii::deallog.depth_console(5);
      {
          std::cout << "Solving with Q2 elements, global refinement"
          << std::endl
          << "============================================="
          << std::endl
          << std::endl;
          Step12::AdvectionProblem<2> dgmethod(Step12::AdvectionProblem<2>::global);
          dgmethod.run();
      }

      {
          std::cout << "Solving with Q2 elements, adaptive refinement"
          << std::endl
          << "============================================="
          << std::endl
          << std::endl;
          Step12::AdvectionProblem<2> dgmethod(Step12::AdvectionProblem<2>::adaptive);
          dgmethod.run();
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
