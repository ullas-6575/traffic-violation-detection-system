<?php

namespace Database\Seeders;

use App\Models\User;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class DatabaseSeeder extends Seeder
{
    use WithoutModelEvents;

    /**
     * Seed the application's database.
     */
    public function run(): void
    {
        // User::factory(10)->create();

        User::updateOrCreate(
            ['email' => 'admin@tvs.local'],
            [
                'name' => 'TVS Administrator',
                'password' => 'password',
                'role' => 'admin',
                'email_verified_at' => now(),
            ],
        );

        User::updateOrCreate(
            ['email' => 'officer@tvs.local'],
            [
                'name' => 'Traffic Officer',
                'password' => 'password',
                'role' => 'officer',
                'email_verified_at' => now(),
            ],
        );

        $this->call(ViolationSeeder::class);
    }
}
