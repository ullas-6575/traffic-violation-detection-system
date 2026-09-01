<x-app-layout>
    <x-slot name="header">
        <h2 class="font-semibold text-xl text-gray-800 leading-tight">
            {{ __('Dashboard') }}
        </h2>
    </x-slot>

    <div class="py-12">
        <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                <div class="p-6 text-gray-900">
                    <p class="font-semibold">{{ __('Traffic Violation Dashboard') }}</p>
                    <p class="mt-2 text-sm text-gray-600">
                        Signed in as {{ auth()->user()->name }}
                        <span class="ml-2 rounded-full bg-slate-100 px-2 py-1 text-xs font-semibold uppercase">
                            {{ auth()->user()->role }}
                        </span>
                    </p>
                </div>
            </div>
        </div>
    </div>
</x-app-layout>
