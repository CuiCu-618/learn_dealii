//
// Created by Cu Cui on 2022/1/20.
//

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <iostream>

using namespace dealii;

template <int dim, int fe_degree>
void
test_reference()
{
  Triangulation<dim>        triangulation;
  DoFHandler<dim>           dof_handler(triangulation);
  FE_Q<dim>                 fe(fe_degree);
  AffineConstraints<double> constraints;
  SparseMatrix<double>      system_matrix;
  SparsityPattern           sparsity_pattern;
  Vector<double>            solution;
  Vector<double>            system_rhs;


  // setup
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  dof_handler.distribute_dofs(fe);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  // assemble
  const QGauss<dim>  quadrature_formula(fe.degree + 1);
  FEValues<dim>      fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            {
              for (const unsigned int j : fe_values.dof_indices())
                cell_matrix(i, j) +=
                  (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                   fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                   fe_values.JxW(q_index));           // dx

              cell_rhs(i) += (1. *                                // f(x)
                              fe_values.shape_value(i, q_index) * // phi_i(x_q)
                              fe_values.JxW(q_index));            // dx
            }
        }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

  // output
  // system_matrix.print_formatted(std::cout);
  // system_rhs.print(std::cout);
}

template <int dim, int fe_degree>
void
test_mf()
{
  Triangulation<dim>        triangulation;
  FE_Q<dim>                 fe(fe_degree);
  DoFHandler<dim>           dof_handler(triangulation);
  MappingQ1<dim>            mapping;
  AffineConstraints<double> constraints;
  Vector<double>            system_rhs;

  //  setup
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  dof_handler.distribute_dofs(fe);
  system_rhs.reinit(dof_handler.n_dofs());

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, double>::AdditionalData::none;
  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points);
  std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(
    new MatrixFree<dim, double>());
  system_mf_storage->reinit(mapping,
                            dof_handler,
                            constraints,
                            QGauss<1>(fe.degree + 1),
                            additional_data);
  // assemble rhs
  system_rhs = 0;
  {
  Timer time;
  FEEvaluation<dim, fe_degree> phi(*system_mf_storage);
  for (unsigned int cell = 0; cell < system_mf_storage->n_cell_batches();
       ++cell)
    {
      phi.reinit(cell);
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(make_vectorized_array<double>(1.0), q);
      phi.integrate(EvaluationFlags::values);
      phi.distribute_local_to_global(system_rhs);
    }
  system_rhs.compress(VectorOperation::add);
  std::cout << "Assemble right hand side   (CPU/wall) " << time.cpu_time()
            << "s/" << time.wall_time() << "s" << std::endl;
  }
  // system_rhs.print(std::cout);

  std::cout << "Number of cells       : " << triangulation.n_active_cells()
            << std::endl
            << "Number of cell batches: " << system_mf_storage->n_cell_batches()
            << std::endl;
  const unsigned int n_vect_doubles = VectorizedArray<double>::size();
  const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;
  std::cout << "Vectorization over " << n_vect_doubles
            << " doubles = " << n_vect_bits << " bits ("
            << Utilities::System::get_current_vectorization_level() << ")"
            << std::endl;
}

int
main()
{
  test_reference<2, 1>();
  test_mf<2, 1>();
  return 0;
}