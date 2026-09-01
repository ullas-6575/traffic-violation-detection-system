<?php

use App\Http\Controllers\ProfileController;
use App\Http\Controllers\ViolationController;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return auth()->check()
        ? redirect()->route('dashboard')
        : redirect()->route('login');
});

Route::get('/dashboard', function () {
    return view('dashboard');
})->middleware(['auth', 'role:admin,officer'])->name('dashboard');

Route::view('/admin', 'admin.index')
    ->middleware(['auth', 'role:admin'])
    ->name('admin.index');

Route::get('/violations', [ViolationController::class, 'index'])
    ->middleware(['auth', 'role:admin,officer'])
    ->name('violations.index');

Route::get('/violations/{violation}', [ViolationController::class, 'show'])
    ->middleware(['auth', 'role:admin,officer'])
    ->name('violations.show');

Route::patch('/violations/{violation}', [ViolationController::class, 'update'])
    ->middleware(['auth', 'role:admin,officer'])
    ->name('violations.update');

Route::middleware('auth')->group(function () {
    Route::get('/profile', [ProfileController::class, 'edit'])->name('profile.edit');
    Route::patch('/profile', [ProfileController::class, 'update'])->name('profile.update');
    Route::delete('/profile', [ProfileController::class, 'destroy'])->name('profile.destroy');
});

require __DIR__.'/auth.php';
